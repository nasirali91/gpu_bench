#include <iostream>
#include <fstream>
#include <thread>
#include <chrono>
#include <cassert>
#include <vector>
#include <string>
#include <mutex>
#include <atomic>
#include <algorithm>
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

struct TimestampedMetrics
{
    unsigned long long timestamp_ms;
    unsigned int power = 0;
    unsigned int sm_utilization = 0;
    unsigned int mem_utilization = 0;
    unsigned int sm_clock = 0;
    unsigned int mem_clock = 0;
    double energy_joules = 0.0;
    std::string event;
};

// Append event to the closest timestamped metric
void append_kernel_event_to_nearest(std::vector<TimestampedMetrics>& metrics, std::mutex& mtx,
                                    unsigned long long timestamp, const std::string& event)
{
    std::lock_guard<std::mutex> lock(mtx);
    if (metrics.empty()) return;
    auto it = std::min_element(metrics.begin(), metrics.end(),
        [timestamp](const TimestampedMetrics& a, const TimestampedMetrics& b) {
            return std::abs((long long)a.timestamp_ms - (long long)timestamp)
                 < std::abs((long long)b.timestamp_ms - (long long)timestamp);
        });
    if (!it->event.empty())
        it->event += " | ";
    it->event += event;
}

void measure_metrics(int sampling_period, int num_runs, int delay_between_runs, int scale, float percent)
{
    nvmlDevice_t device;
    checkNvmlError(nvmlInit(), "Failed to initialize NVML");
    checkNvmlError(nvmlDeviceGetHandleByIndex(0, &device), "Failed to get NVML device handle");
    // 
    std::atomic<int> sampling_period_atomic(sampling_period);
    auto app_start_time = std::chrono::high_resolution_clock::now();
    std::vector<TimestampedMetrics> all_metrics;
    std::mutex metrics_mutex;

    unsigned long long initial_energy = 0;
    checkNvmlError(nvmlDeviceGetTotalEnergyConsumption(device, &initial_energy), "Failed to get initial energy usage");

    std::atomic<bool> monitoring_active(true);
    std::thread monitoring_thread([&]()
    {
        auto next_sample_time = std::chrono::high_resolution_clock::now();
        while (monitoring_active.load()) {
            unsigned int power;
            nvmlUtilization_t utilization;
            unsigned int sm_clock, mem_clock;
            unsigned long long current_energy = 0;

            checkNvmlError(nvmlDeviceGetPowerUsage(device, &power), "Failed to get power usage");
            checkNvmlError(nvmlDeviceGetUtilizationRates(device, &utilization), "Failed to get utilization rates");
            checkNvmlError(nvmlDeviceGetClockInfo(device, NVML_CLOCK_SM, &sm_clock), "Failed to get SM clock");
            checkNvmlError(nvmlDeviceGetClockInfo(device, NVML_CLOCK_MEM, &mem_clock), "Failed to get memory clock");
            checkNvmlError(nvmlDeviceGetTotalEnergyConsumption(device, &current_energy), "Failed to get energy usage");

            auto current_time = std::chrono::high_resolution_clock::now();
            auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                current_time - app_start_time).count();

            TimestampedMetrics metrics;
            metrics.timestamp_ms = elapsed_ms;
            metrics.power = power / 1000;
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

            int current_period = sampling_period_atomic.load();
            next_sample_time += std::chrono::milliseconds(current_period);
            std::this_thread::sleep_until(next_sample_time);

          //  next_sample_time += std::chrono::milliseconds(sampling_period);
         //   std::this_thread::sleep_until(next_sample_time);
        }
    });

    

    // Get device properties and pre-allocate memory
    cudaDeviceProp devProp;
    checkCudaError(cudaGetDeviceProperties(&devProp, 0), "Failed to get device properties");
    int nblocks = static_cast<int>(devProp.multiProcessorCount * percent / 100.0);
    int nthreads = devProp.maxThreadsPerBlock;
    nblocks = (nblocks < 1) ? 1 : nblocks;
    int nsize = nblocks * nthreads;

    float *d_x;
    checkCudaError(cudaMalloc(&d_x, nsize * sizeof(float)), "Failed to allocate device memory");
    std::vector<float> h_x(nsize, 0.0f);
    checkCudaError(cudaMemcpy(d_x, h_x.data(), nsize * sizeof(float), cudaMemcpyHostToDevice), "Failed to copy data to device");

    cudaEvent_t kernel_start, kernel_stop;
    checkCudaError(cudaEventCreate(&kernel_start), "Failed to create start event");
    checkCudaError(cudaEventCreate(&kernel_stop), "Failed to create stop event");

    unsigned int delay = 44800;
    auto scheduled_next_start = std::chrono::high_resolution_clock::now();

    for (int run = 0; run < num_runs; ++run)
    {
        // Wait until scheduled start time
        auto now = std::chrono::high_resolution_clock::now();
        if (now < scheduled_next_start)
            std::this_thread::sleep_until(scheduled_next_start);

        checkCudaError(cudaDeviceSynchronize(), "Failed to synchronize before kernel");

        // Record host-side kernel start timestamp
        auto host_kernel_start = std::chrono::high_resolution_clock::now();
        auto kernel_start_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            host_kernel_start - app_start_time).count();

        checkCudaError(cudaEventRecord(kernel_start), "Failed to record start event");
        kernel<<<nblocks, nthreads>>>(d_x, delay * scale);
        checkCudaError(cudaEventRecord(kernel_stop), "Failed to record stop event");
        checkCudaError(cudaEventSynchronize(kernel_stop), "Failed to synchronize stop event");

        // Record host-side kernel end timestamp
        auto host_kernel_end = std::chrono::high_resolution_clock::now();
        auto kernel_end_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            host_kernel_end - app_start_time).count();

        float kernel_latency_ms;
        checkCudaError(cudaEventElapsedTime(&kernel_latency_ms, kernel_start, kernel_stop), "Failed to calculate elapsed time");

        // Add kernel events to the closest metrics
        append_kernel_event_to_nearest(all_metrics, metrics_mutex, kernel_start_ms, "Kernel start");
        append_kernel_event_to_nearest(all_metrics, metrics_mutex, kernel_end_ms, "Kernel end");

        std::cout << "Run " << run + 1
                  << ": Kernel started at " << kernel_start_ms << " ms, "
                  << "finished at " << kernel_end_ms << " ms\n";
        std::cout << "  Kernel Latency (CUDA event): " << kernel_latency_ms << " ms\n";

        scheduled_next_start = std::chrono::high_resolution_clock::now() + std::chrono::milliseconds(delay_between_runs);
    }

    checkCudaError(cudaFree(d_x), "Failed to free device memory");
    checkCudaError(cudaEventDestroy(kernel_start), "Failed to destroy start event");
    checkCudaError(cudaEventDestroy(kernel_stop), "Failed to destroy stop event");

    //std::this_thread::sleep_for(std::chrono::seconds(3));

    sampling_period_atomic = 150; // e.g., 150 ms (0.15 second) 
    std::this_thread::sleep_for(std::chrono::seconds(3));
    monitoring_active.store(false);
    if (monitoring_thread.joinable())
        monitoring_thread.join();

    std::ofstream csv_file("metrics.csv");
    csv_file << "Timestamp(ms),Power(W),SM Utilization(%),Memory Utilization(%),SM Clock(MHz),Memory Clock(MHz),Energy(J),Event\n";
    {
        std::lock_guard<std::mutex> lock(metrics_mutex);
        std::sort(all_metrics.begin(), all_metrics.end(),
                  [](const TimestampedMetrics& a, const TimestampedMetrics& b) {
                      return a.timestamp_ms < b.timestamp_ms;
                  });
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
