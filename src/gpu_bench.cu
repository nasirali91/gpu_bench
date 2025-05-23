#include <iostream>
#include <fstream>
#include <thread>
#include <chrono>
#include <cassert>
#include <random>
#include <atomic>
#include <mutex>
#include <vector>
#include <string>
#include <nvml.h>
#include <cuda_runtime.h>

__global__ void kernel(float *x, int nit)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
#pragma unroll
    for (int i = 0; i < nit; i++)
    {
        x[tid] = x[tid] * 2 + 2;
        x[tid] = x[tid] / 2 - 1;
    }
}

void checkCudaError(cudaError_t err, const char *msg)
{
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA Error: " << msg << " - " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}

void checkNvmlError(nvmlReturn_t result, const char *msg)
{
    if (result != NVML_SUCCESS)
    {
        std::cerr << "NVML Error: " << msg << " - " << nvmlErrorString(result) << std::endl;
        exit(EXIT_FAILURE);
    }
}

// Structure to hold NVML metrics with timestamp
struct TimestampedMetrics
{
    unsigned long long timestamp_ms; // Timestamp in milliseconds since start
    unsigned int power = 0;
    unsigned int sm_utilization = 0;
    unsigned int mem_utilization = 0;
    unsigned int sm_clock = 0;
    unsigned int mem_clock = 0;
    double energy_joules = 0.0;
    std::string event;
};

void measure_metrics(int sampling_period, int num_runs, int delay_between_runs, int scale, float percent)
{
    nvmlDevice_t device;

    checkNvmlError(nvmlInit(), "Failed to initialize NVML");
    checkNvmlError(nvmlDeviceGetHandleByIndex(0, &device), "Failed to get NVML device handle");

    // Record application start time
    auto app_start_time = std::chrono::high_resolution_clock::now();

    // Vector to store all metrics with timestamps
    std::vector<TimestampedMetrics> all_metrics;
    std::mutex metrics_mutex;

    // Get initial energy reading to use as baseline
    unsigned long long initial_energy = 0;
    try {
        checkNvmlError(nvmlDeviceGetTotalEnergyConsumption(device, &initial_energy), "Failed to get initial energy usage");
    } catch (const std::exception& e) {
        std::cerr << "Warning: Energy measurement not available: " << e.what() << std::endl;
    }

    // Start a thread to continuously measure NVML metrics
    std::atomic<bool> monitoring_active(true);
    std::thread monitoring_thread([&]()
    {
        try {
            auto next_sample_time = std::chrono::high_resolution_clock::now();
            while (monitoring_active.load()) {
                // Collect metrics
                unsigned int power;
                nvmlUtilization_t utilization;
                unsigned int sm_clock, mem_clock;
                unsigned long long current_energy = 0;

                checkNvmlError(nvmlDeviceGetPowerUsage(device, &power), "Failed to get power usage");
                checkNvmlError(nvmlDeviceGetUtilizationRates(device, &utilization), "Failed to get utilization rates");
                checkNvmlError(nvmlDeviceGetClockInfo(device, NVML_CLOCK_SM, &sm_clock), "Failed to get SM clock");
                checkNvmlError(nvmlDeviceGetClockInfo(device, NVML_CLOCK_MEM, &mem_clock), "Failed to get memory clock");
                try {
                    checkNvmlError(nvmlDeviceGetTotalEnergyConsumption(device, &current_energy), "Failed to get energy usage");
                } catch (...) {
                    current_energy = 0;
                }

                auto current_time = std::chrono::high_resolution_clock::now();
                auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                    current_time - app_start_time).count();

                TimestampedMetrics metrics;
                metrics.timestamp_ms = elapsed_ms;
                metrics.power = power / 1000; // convert from milliwatts to watts
                metrics.sm_utilization = utilization.gpu;
                metrics.mem_utilization = utilization.memory;
                metrics.sm_clock = sm_clock;
                metrics.mem_clock = mem_clock;
                metrics.energy_joules = (current_energy - initial_energy) / 1000.0;
                metrics.event = "";

                {
                    std::lock_guard<std::mutex> lock(metrics_mutex);
                    all_metrics.push_back(metrics);
                }

                next_sample_time += std::chrono::milliseconds(sampling_period);
                std::this_thread::sleep_until(next_sample_time);
            }
        } catch (const std::exception& e) {
            std::cerr << "Error in monitoring thread: " << e.what() << std::endl;
        }
    });

    // Create CUDA events for kernel timing
    cudaEvent_t kernel_start, kernel_stop;
    checkCudaError(cudaEventCreate(&kernel_start), "Failed to create start event");
    checkCudaError(cudaEventCreate(&kernel_stop), "Failed to create stop event");

    auto last_kernel_end_time = std::chrono::high_resolution_clock::now();

    for (int run = 0; run < num_runs; ++run)
    {

        std::cout << "Starting run " << run + 1 << " of " << num_runs << std::endl;
        auto this_kernel_start_time = std::chrono::high_resolution_clock::now();
        if (run > 0) {
            auto delay_ms = std::chrono::duration<double, std::milli>(this_kernel_start_time - last_kernel_end_time).count();
            std::cout << "Actual delay between run " << run << " and " << (run+1) << ": " << delay_ms << " ms" << std::endl;
        }
        int nblocks, nthreads, nsize;

        cudaDeviceProp devProp;
        checkCudaError(cudaGetDeviceProperties(&devProp, 0), "Failed to get device properties");

        nblocks = static_cast<int>(devProp.multiProcessorCount * percent / 100.0);
        nthreads = devProp.maxThreadsPerBlock;
        nblocks = (nblocks < 1) ? 1 : nblocks;
        nsize = nblocks * nthreads;

        float *d_x;
        checkCudaError(cudaMalloc(&d_x, nsize * sizeof(float)), "Failed to allocate device memory");

        std::vector<float> h_x(nsize);

        checkCudaError(cudaMemcpy(d_x, h_x.data(), nsize * sizeof(float), cudaMemcpyHostToDevice), "Failed to copy data to device");

        // Mark kernel start 
        auto kernel_start_time = std::chrono::high_resolution_clock::now();
        auto kernel_start_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                              kernel_start_time - app_start_time).count();
        std::cout << "Kernel " << run + 1 << " starting at " << kernel_start_ms << " ms" << std::endl;

        // --- Kernel start marker with NVML values ---
        {
            unsigned int power;
            nvmlUtilization_t utilization;
            unsigned int sm_clock, mem_clock;
            unsigned long long current_energy = 0;

            checkNvmlError(nvmlDeviceGetPowerUsage(device, &power), "Failed to get power usage");
            checkNvmlError(nvmlDeviceGetUtilizationRates(device, &utilization), "Failed to get utilization rates");
            checkNvmlError(nvmlDeviceGetClockInfo(device, NVML_CLOCK_SM, &sm_clock), "Failed to get SM clock");
            checkNvmlError(nvmlDeviceGetClockInfo(device, NVML_CLOCK_MEM, &mem_clock), "Failed to get memory clock");
            try {
                checkNvmlError(nvmlDeviceGetTotalEnergyConsumption(device, &current_energy), "Failed to get energy usage");
            } catch (...) {
                current_energy = 0;
            }

            TimestampedMetrics start_marker;
            start_marker.timestamp_ms = kernel_start_ms;
            start_marker.power = power / 1000;
            start_marker.sm_utilization = utilization.gpu;
            start_marker.mem_utilization = utilization.memory;
            start_marker.sm_clock = sm_clock;
            start_marker.mem_clock = mem_clock;
            start_marker.energy_joules = (current_energy - initial_energy) / 1000.0;
            start_marker.event = "Kernel start";
            all_metrics.push_back(start_marker);
        }

        // Record start event
        checkCudaError(cudaEventRecord(kernel_start), "Failed to record start event");

        // Launch the kernel
        unsigned int delay = 43900;
        kernel<<<nblocks, nthreads>>>(d_x, delay * scale);

        // Record stop event
        checkCudaError(cudaEventRecord(kernel_stop), "Failed to record stop event");
        checkCudaError(cudaEventSynchronize(kernel_stop), "Failed to synchronize stop event");

        // Calculate kernel latency
        float kernel_latency_ms;
        checkCudaError(cudaEventElapsedTime(&kernel_latency_ms, kernel_start, kernel_stop), "Failed to calculate elapsed time");

        checkCudaError(cudaDeviceSynchronize(), "Kernel execution failed");

        // Log kernel end time
        auto kernel_end_time = std::chrono::high_resolution_clock::now();
        auto kernel_end_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                              kernel_end_time - app_start_time).count();
        std::cout << "Kernel " << run + 1 << " finished at " << kernel_end_ms << " ms" << std::endl;

        std::cout << "Run " << run + 1 << " - Kernel Latency: " << kernel_latency_ms << " ms" << std::endl;

        // --- Kernel end marker with NVML values ---
        {
            unsigned int power;
            nvmlUtilization_t utilization;
            unsigned int sm_clock, mem_clock;
            unsigned long long current_energy = 0;

            checkNvmlError(nvmlDeviceGetPowerUsage(device, &power), "Failed to get power usage");
            checkNvmlError(nvmlDeviceGetUtilizationRates(device, &utilization), "Failed to get utilization rates");
            checkNvmlError(nvmlDeviceGetClockInfo(device, NVML_CLOCK_SM, &sm_clock), "Failed to get SM clock");
            checkNvmlError(nvmlDeviceGetClockInfo(device, NVML_CLOCK_MEM, &mem_clock), "Failed to get memory clock");
            try {
                checkNvmlError(nvmlDeviceGetTotalEnergyConsumption(device, &current_energy), "Failed to get energy usage");
            } catch (...) {
                current_energy = 0;
            }

            TimestampedMetrics end_marker;
            end_marker.timestamp_ms = kernel_end_ms;
            end_marker.power = power / 1000;
            end_marker.sm_utilization = utilization.gpu;
            end_marker.mem_utilization = utilization.memory;
            end_marker.sm_clock = sm_clock;
            end_marker.mem_clock = mem_clock;
            end_marker.energy_joules = (current_energy - initial_energy) / 1000.0;
            end_marker.event = "Kernel end";
            all_metrics.push_back(end_marker);
        }

        // Free memory before the next iteration
        checkCudaError(cudaFree(d_x), "Failed to free device memory");

        // Wait before starting the next run
        if (run < num_runs - 1)
        {
            auto after_cleanup_time = std::chrono::high_resolution_clock::now();
            auto elapsed_since_kernel_end = std::chrono::duration_cast<std::chrono::milliseconds>(
                after_cleanup_time - kernel_end_time).count();

            int sleep_time = delay_between_runs - static_cast<int>(elapsed_since_kernel_end);
            if (sleep_time > 0)
            {
                std::this_thread::sleep_for(std::chrono::milliseconds(sleep_time));
            }
            else
            {
                std::cout << "No wait needed before next run (delay_between_runs already exceeded by " 
                          << -sleep_time << " ms)." << std::endl;
            }
        }
    }

    // Cleanup events
    checkCudaError(cudaEventDestroy(kernel_start), "Failed to destroy start event");
    checkCudaError(cudaEventDestroy(kernel_stop), "Failed to destroy stop event");

    // Continue monitoring after the last kernel
    std::this_thread::sleep_for(std::chrono::seconds(3));

    // Stop the monitoring thread
    monitoring_active.store(false);
    if (monitoring_thread.joinable())
    {
        monitoring_thread.join();
    }

    // Write all metrics to CSV
    std::ofstream csv_file("metrics.csv");
    csv_file << "Timestamp(ms),Power(W),SM Utilization(%),Memory Utilization(%),SM Clock(MHz),Memory Clock(MHz),Energy(J),Event\n";

    {
        std::lock_guard<std::mutex> lock(metrics_mutex);
        for (const auto &metric : all_metrics)
        {
            csv_file << metric.timestamp_ms << ","
                     << metric.power << ","
                     << metric.sm_utilization << ","
                     << metric.mem_utilization << ","
                     << metric.sm_clock << ","
                     << metric.mem_clock << ","
                     << metric.energy_joules << ","
                     << metric.event << "\n";
        }
    }

    csv_file.close();
    checkNvmlError(nvmlShutdown(), "Failed to shutdown NVML");
}

int main(int argc, const char **argv)
{
    if (argc != 6)
    {
        std::cerr << "Usage: " << argv[0] << " <sampling_period> <num_runs> <delay_between_runs> <scale> <percent>\n";
        return EXIT_FAILURE;
    }

    int sampling_period = std::stoi(argv[1]);
    int num_runs = std::stoi(argv[2]);
    int delay_between_runs = std::stoi(argv[3]);
    int scale = std::stoi(argv[4]);
    float percent = std::stof(argv[5]);

    assert(sampling_period > 0 && "Sampling period must be positive");
    assert(num_runs > 0 && "Number of runs must be positive");
    assert(delay_between_runs >= 0 && "Delay between runs must be non-negative");
    assert(scale > 0 && "Scale must be positive");
    assert(percent > 0 && percent <= 100 && "Percent must be between 1 and 100");

    measure_metrics(sampling_period, num_runs, delay_between_runs, scale, percent);

    return EXIT_SUCCESS;
}
