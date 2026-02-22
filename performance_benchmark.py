#!/usr/bin/env python3
"""
Computational Performance Benchmark for YOLO11 Masking Techniques

This script measures the computational cost of various masking operations 
to address the paper's need for latency and throughput analysis.

Author: Performance Analysis Team
Date: 2024
"""

import cv2
import numpy as np
import time
import json
import os
import sys
from statistics import mean, stdev
from ultralytics import YOLO
import psutil
import gc
from pathlib import Path

class MaskingPerformanceBenchmark:
    def __init__(self, model_path='best_yolo11.pt', test_video='test_video.mp4'):
        """Initialize benchmark with YOLO model and test video"""
        self.model_path = model_path
        self.test_video = test_video
        self.model = None
        self.class_names = [
            'trak', 'cyclist', 'bike', 'tempo', 'car', 'zeep', 'toto',
            'e-rickshaw', 'auto-rickshaw', 'bus', 'van', 'cycle-rickshaw',
            'person', 'taxi'
        ]
        self.confidence_threshold = 0.5
        self.results = {}
        
    def load_model(self):
        """Load YOLO model and measure loading time"""
        print("Loading YOLO11 model...")
        start_time = time.time()
        self.model = YOLO(self.model_path)
        load_time = time.time() - start_time
        print(f"Model loaded in {load_time:.3f} seconds")
        return load_time

    def get_system_info(self):
        """Collect system information for context"""
        import platform
        import torch
        
        info = {
            'platform': platform.platform(),
            'processor': platform.processor(),
            'python_version': platform.python_version(),
            'cpu_count': psutil.cpu_count(),
            'memory_gb': round(psutil.virtual_memory().total / (1024**3), 2),
            'cuda_available': torch.cuda.is_available(),
            'torch_version': torch.__version__
        }
        
        if torch.cuda.is_available():
            info['gpu_name'] = torch.cuda.get_device_name(0)
            info['gpu_memory_gb'] = round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2)
        
        return info

    def measure_yolo_inference(self, frame, iterations=10):
        """Measure pure YOLO inference time"""
        times = []
        for _ in range(iterations):
            start_time = time.perf_counter()
            results = self.model(frame, conf=self.confidence_threshold)
            end_time = time.perf_counter()
            times.append(end_time - start_time)
        
        return {
            'mean_time': mean(times),
            'std_time': stdev(times) if len(times) > 1 else 0,
            'min_time': min(times),
            'max_time': max(times),
            'times': times
        }

    def bw_mask_benchmark(self, frame, selected_object='car', iterations=10):
        """Benchmark Black & White masking technique"""
        times = []
        memory_usage = []
        
        for _ in range(iterations):
            gc.collect()  # Clean memory before measurement
            process = psutil.Process(os.getpid())
            start_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            start_time = time.perf_counter()
            
            # Road detection
            road_start = time.perf_counter()
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            lower_gray = np.array([0, 0, 50])
            upper_gray = np.array([180, 30, 200])
            road_mask = cv2.inRange(hsv, lower_gray, upper_gray)
            kernel = np.ones((5,5), np.uint8)
            road_mask = cv2.morphologyEx(road_mask, cv2.MORPH_CLOSE, kernel)
            road_time = time.perf_counter() - road_start
            
            # Object detection
            detection_start = time.perf_counter()
            results = self.model(frame, conf=self.confidence_threshold)
            detection_time = time.perf_counter() - detection_start
            
            # Mask creation
            mask_start = time.perf_counter()
            object_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            for det in results[0].boxes:
                x1,y1,x2,y2 = det.xyxy[0].cpu().numpy().astype(int)
                label = self.class_names[int(det.cls[0].cpu().item())]
                if label == selected_object:
                    cv2.rectangle(object_mask, (x1,y1), (x2,y2), 255, -1)
            
            final_mask = cv2.bitwise_or(road_mask, object_mask)
            mask_time = time.perf_counter() - mask_start
            
            # Image processing
            process_start = time.perf_counter()
            desat = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            desat = cv2.cvtColor(desat, cv2.COLOR_GRAY2BGR)
            result = frame.copy()
            result[final_mask == 0] = desat[final_mask == 0]
            process_time = time.perf_counter() - process_start
            
            end_time = time.perf_counter()
            total_time = end_time - start_time
            
            end_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_usage.append(end_memory - start_memory)
            times.append(total_time)
        
        return {
            'technique': 'Black & White',
            'mean_time': mean(times),
            'std_time': stdev(times) if len(times) > 1 else 0,
            'min_time': min(times),
            'max_time': max(times),
            'mean_memory_mb': mean(memory_usage),
            'breakdown': {
                'road_detection': road_time,
                'yolo_inference': detection_time,
                'mask_creation': mask_time,
                'image_processing': process_time
            }
        }

    def complete_blackout_benchmark(self, frame, selected_object='car', iterations=10):
        """Benchmark Complete Blackout technique"""
        times = []
        memory_usage = []
        
        for _ in range(iterations):
            gc.collect()
            process = psutil.Process(os.getpid())
            start_memory = process.memory_info().rss / 1024 / 1024
            
            start_time = time.perf_counter()
            
            results = self.model(frame, conf=self.confidence_threshold)
            result = np.zeros_like(frame)
            
            for det in results[0].boxes:
                x1,y1,x2,y2 = det.xyxy[0].cpu().numpy().astype(int)
                label = self.class_names[int(det.cls[0].cpu().item())]
                if label == selected_object:
                    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
                    cv2.rectangle(mask, (x1,y1), (x2,y2), 255, -1)
                    obj = cv2.bitwise_and(frame, frame, mask=mask)
                    result = cv2.add(result, obj)
            
            end_time = time.perf_counter()
            end_memory = process.memory_info().rss / 1024 / 1024
            
            times.append(end_time - start_time)
            memory_usage.append(end_memory - start_memory)
        
        return {
            'technique': 'Complete Blackout',
            'mean_time': mean(times),
            'std_time': stdev(times) if len(times) > 1 else 0,
            'min_time': min(times),
            'max_time': max(times),
            'mean_memory_mb': mean(memory_usage)
        }

    def blur_blackout_benchmark(self, frame, selected_object='car', iterations=10):
        """Benchmark Blur Blackout technique"""
        times = []
        memory_usage = []
        
        for _ in range(iterations):
            gc.collect()
            process = psutil.Process(os.getpid())
            start_memory = process.memory_info().rss / 1024 / 1024
            
            start_time = time.perf_counter()
            
            results = self.model(frame, conf=self.confidence_threshold)
            mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            
            for det in results[0].boxes:
                x1,y1,x2,y2 = det.xyxy[0].cpu().numpy().astype(int)
                label = self.class_names[int(det.cls[0].cpu().item())]
                if label == selected_object:
                    cv2.rectangle(mask, (x1,y1), (x2,y2), 255, -1)
            
            inverse = cv2.bitwise_not(mask)
            blurred = cv2.GaussianBlur(frame, (35,35), 0)
            darkened = cv2.addWeighted(frame, 0.3, np.zeros_like(frame), 0.7, 0)
            background = cv2.addWeighted(blurred, 0.7, darkened, 0.3, 0)
            result = frame.copy()
            result[inverse == 255] = background[inverse == 255]
            
            end_time = time.perf_counter()
            end_memory = process.memory_info().rss / 1024 / 1024
            
            times.append(end_time - start_time)
            memory_usage.append(end_memory - start_memory)
        
        return {
            'technique': 'Blur Blackout',
            'mean_time': mean(times),
            'std_time': stdev(times) if len(times) > 1 else 0,
            'min_time': min(times),
            'max_time': max(times),
            'mean_memory_mb': mean(memory_usage)
        }

    def complete_inpainting_benchmark(self, frame, selected_object='car', iterations=10):
        """Benchmark Complete Inpainting technique - most computationally intensive"""
        times = []
        memory_usage = []
        
        for _ in range(iterations):
            gc.collect()
            process = psutil.Process(os.getpid())
            start_memory = process.memory_info().rss / 1024 / 1024
            
            start_time = time.perf_counter()
            
            results = self.model(frame, conf=self.confidence_threshold)
            mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            
            for det in results[0].boxes:
                x1,y1,x2,y2 = det.xyxy[0].cpu().numpy().astype(int)
                label = self.class_names[int(det.cls[0].cpu().item())]
                if label == selected_object:
                    cv2.rectangle(mask, (x1,y1), (x2,y2), 255, -1)
            
            combined = cv2.bitwise_not(mask)
            
            # Two inpainting methods - computationally expensive
            inpaint_start = time.perf_counter()
            inpaint_ns = cv2.inpaint(frame, combined, 3, cv2.INPAINT_NS)
            inpaint_tl = cv2.inpaint(frame, combined, 3, cv2.INPAINT_TELEA)
            result = cv2.addWeighted(inpaint_ns, 0.5, inpaint_tl, 0.5, 0)
            inpaint_time = time.perf_counter() - inpaint_start
            
            end_time = time.perf_counter()
            end_memory = process.memory_info().rss / 1024 / 1024
            
            times.append(end_time - start_time)
            memory_usage.append(end_memory - start_memory)
        
        return {
            'technique': 'Complete Inpainting',
            'mean_time': mean(times),
            'std_time': stdev(times) if len(times) > 1 else 0,
            'min_time': min(times),
            'max_time': max(times),
            'mean_memory_mb': mean(memory_usage),
            'inpainting_time': inpaint_time
        }

    def gaussian_blur_benchmark(self, frame, selected_object='car', iterations=10):
        """Benchmark Gaussian Blur technique"""
        times = []
        memory_usage = []
        
        for _ in range(iterations):
            gc.collect()
            process = psutil.Process(os.getpid())
            start_memory = process.memory_info().rss / 1024 / 1024
            
            start_time = time.perf_counter()
            
            results = self.model(frame, conf=self.confidence_threshold)
            mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            
            for det in results[0].boxes:
                x1,y1,x2,y2 = det.xyxy[0].cpu().numpy().astype(int)
                label = self.class_names[int(det.cls[0].cpu().item())]
                if label == selected_object:
                    cv2.rectangle(mask, (x1,y1), (x2,y2), 255, -1)
            
            blurred = cv2.GaussianBlur(frame, (21,21), 0)
            result = frame.copy()
            result[mask == 0] = blurred[mask == 0]
            
            end_time = time.perf_counter()
            end_memory = process.memory_info().rss / 1024 / 1024
            
            times.append(end_time - start_time)
            memory_usage.append(end_memory - start_memory)
        
        return {
            'technique': 'Gaussian Blur',
            'mean_time': mean(times),
            'std_time': stdev(times) if len(times) > 1 else 0,
            'min_time': min(times),
            'max_time': max(times),
            'mean_memory_mb': mean(memory_usage)
        }

    def object_inpainting_benchmark(self, frame, selected_object='car', iterations=10):
        """Benchmark Object Inpainting technique"""
        times = []
        memory_usage = []
        
        for _ in range(iterations):
            gc.collect()
            process = psutil.Process(os.getpid())
            start_memory = process.memory_info().rss / 1024 / 1024
            
            start_time = time.perf_counter()
            
            results = self.model(frame, conf=self.confidence_threshold)
            mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            
            for det in results[0].boxes:
                x1,y1,x2,y2 = det.xyxy[0].cpu().numpy().astype(int)
                label = self.class_names[int(det.cls[0].cpu().item())]
                if label != selected_object:
                    cv2.rectangle(mask, (x1,y1), (x2,y2), 255, -1)
            
            result = cv2.inpaint(frame, mask, 3, cv2.INPAINT_TELEA)
            
            end_time = time.perf_counter()
            end_memory = process.memory_info().rss / 1024 / 1024
            
            times.append(end_time - start_time)
            memory_usage.append(end_memory - start_memory)
        
        return {
            'technique': 'Object Inpainting',
            'mean_time': mean(times),
            'std_time': stdev(times) if len(times) > 1 else 0,
            'min_time': min(times),
            'max_time': max(times),
            'mean_memory_mb': mean(memory_usage)
        }

    def selective_blur_benchmark(self, frame, selected_object='car', iterations=10):
        """Benchmark Selective Blur technique"""
        times = []
        memory_usage = []
        
        for _ in range(iterations):
            gc.collect()
            process = psutil.Process(os.getpid())
            start_memory = process.memory_info().rss / 1024 / 1024
            
            start_time = time.perf_counter()
            
            results = self.model(frame, conf=self.confidence_threshold)
            mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            
            for det in results[0].boxes:
                x1,y1,x2,y2 = det.xyxy[0].cpu().numpy().astype(int)
                label = self.class_names[int(det.cls[0].cpu().item())]
                if label != selected_object:
                    cv2.rectangle(mask, (x1,y1), (x2,y2), 255, -1)
            
            replacement = cv2.medianBlur(frame, 15)
            result = frame.copy()
            result[mask == 255] = replacement[mask == 255]
            
            end_time = time.perf_counter()
            end_memory = process.memory_info().rss / 1024 / 1024
            
            times.append(end_time - start_time)
            memory_usage.append(end_memory - start_memory)
        
        return {
            'technique': 'Selective Blur',
            'mean_time': mean(times),
            'std_time': stdev(times) if len(times) > 1 else 0,
            'min_time': min(times),
            'max_time': max(times),
            'mean_memory_mb': mean(memory_usage)
        }

    def run_video_throughput_test(self, max_frames=100):
        """Test video processing throughput for each technique"""
        if not os.path.exists(self.test_video):
            print(f"Warning: Test video {self.test_video} not found. Using synthetic frames.")
            # Create synthetic test frames
            test_frames = [np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8) for _ in range(10)]
        else:
            cap = cv2.VideoCapture(self.test_video)
            test_frames = []
            frame_count = 0
            
            while cap.isOpened() and frame_count < max_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                test_frames.append(frame)
                frame_count += 1
            cap.release()
        
        if not test_frames:
            print("No test frames available")
            return {}
        
        print(f"Testing throughput with {len(test_frames)} frames...")
        
        techniques = {
            'Black & White': self.bw_mask_benchmark,
            'Complete Blackout': self.complete_blackout_benchmark,
            'Blur Blackout': self.blur_blackout_benchmark,
            'Complete Inpainting': self.complete_inpainting_benchmark,
            'Gaussian Blur': self.gaussian_blur_benchmark,
            'Object Inpainting': self.object_inpainting_benchmark,
            'Selective Blur': self.selective_blur_benchmark
        }
        
        throughput_results = {}
        
        for tech_name, benchmark_func in techniques.items():
            print(f"Testing {tech_name}...")
            
            total_time = 0
            processed_frames = 0
            
            for frame in test_frames:
                start = time.perf_counter()
                try:
                    result = benchmark_func(frame, iterations=1)
                    end = time.perf_counter()
                    total_time += (end - start)
                    processed_frames += 1
                except Exception as e:
                    print(f"Error in {tech_name}: {e}")
                    continue
            
            if processed_frames > 0:
                fps = processed_frames / total_time
                avg_time_per_frame = total_time / processed_frames
                
                throughput_results[tech_name] = {
                    'fps': fps,
                    'avg_time_per_frame_ms': avg_time_per_frame * 1000,
                    'total_time': total_time,
                    'processed_frames': processed_frames
                }
        
        return throughput_results

    def run_comprehensive_benchmark(self):
        """Run comprehensive performance benchmark"""
        print("=" * 60)
        print("YOLO11 Masking Techniques Performance Benchmark")
        print("=" * 60)
        
        # Load model
        model_load_time = self.load_model()
        
        # Get system info
        system_info = self.get_system_info()
        print(f"System: {system_info['platform']}")
        print(f"CPU: {system_info['processor']} ({system_info['cpu_count']} cores)")
        print(f"Memory: {system_info['memory_gb']} GB")
        if system_info['cuda_available']:
            print(f"GPU: {system_info['gpu_name']} ({system_info['gpu_memory_gb']} GB)")
        print("-" * 60)
        
        # Create test frame if video not available
        if os.path.exists(self.test_video):
            cap = cv2.VideoCapture(self.test_video)
            ret, test_frame = cap.read()
            cap.release()
            if not ret:
                test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        else:
            test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # Test YOLO inference baseline
        print("Measuring YOLO11 inference baseline...")
        yolo_baseline = self.measure_yolo_inference(test_frame)
        
        # Test each masking technique
        techniques = {
            'Black & White': self.bw_mask_benchmark,
            'Complete Blackout': self.complete_blackout_benchmark,
            'Blur Blackout': self.blur_blackout_benchmark,
            'Complete Inpainting': self.complete_inpainting_benchmark,
            'Gaussian Blur': self.gaussian_blur_benchmark,
            'Object Inpainting': self.object_inpainting_benchmark,
            'Selective Blur': self.selective_blur_benchmark
        }
        
        technique_results = {}
        
        for tech_name, benchmark_func in techniques.items():
            print(f"Benchmarking {tech_name}...")
            try:
                result = benchmark_func(test_frame)
                technique_results[tech_name] = result
                print(f"  Average time: {result['mean_time']*1000:.2f} ms (±{result['std_time']*1000:.2f})")
                print(f"  Memory usage: {result['mean_memory_mb']:.2f} MB")
            except Exception as e:
                print(f"  Error: {e}")
                technique_results[tech_name] = {'error': str(e)}
        
        # Video throughput test
        print("\nTesting video throughput...")
        throughput_results = self.run_video_throughput_test()
        
        # Compile final results
        self.results = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'system_info': system_info,
            'model_load_time': model_load_time,
            'yolo_baseline': yolo_baseline,
            'technique_benchmarks': technique_results,
            'video_throughput': throughput_results
        }
        
        return self.results

    def generate_report(self, output_file='performance_report.json'):
        """Generate comprehensive performance report"""
        if not self.results:
            print("No results available. Run benchmark first.")
            return
        
        # Save JSON report
        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        # Generate readable summary
        summary_file = output_file.replace('.json', '_summary.txt')
        with open(summary_file, 'w') as f:
            f.write("YOLO11 Masking Techniques Performance Analysis\n")
            f.write("=" * 50 + "\n\n")
            
            f.write("EXECUTIVE SUMMARY\n")
            f.write("-" * 20 + "\n")
            f.write(f"Test Date: {self.results['timestamp']}\n")
            f.write(f"YOLO Model Load Time: {self.results['model_load_time']:.3f} seconds\n")
            f.write(f"YOLO Baseline Inference: {self.results['yolo_baseline']['mean_time']*1000:.2f} ms\n\n")
            
            f.write("TECHNIQUE PERFORMANCE COMPARISON\n")
            f.write("-" * 35 + "\n")
            f.write(f"{'Technique':<20} {'Avg Time (ms)':<15} {'Memory (MB)':<12} {'FPS':<8}\n")
            f.write("-" * 55 + "\n")
            
            for tech_name, result in self.results['technique_benchmarks'].items():
                if 'error' not in result:
                    avg_time = result['mean_time'] * 1000
                    memory = result['mean_memory_mb']
                    fps = self.results['video_throughput'].get(tech_name, {}).get('fps', 0)
                    f.write(f"{tech_name:<20} {avg_time:<15.2f} {memory:<12.2f} {fps:<8.1f}\n")
            
            f.write("\nREAL-TIME APPLICABILITY ANALYSIS\n")
            f.write("-" * 35 + "\n")
            
            realtime_threshold = 33.33  # 30 FPS = 33.33ms per frame
            
            for tech_name, result in self.results['technique_benchmarks'].items():
                if 'error' not in result:
                    avg_time_ms = result['mean_time'] * 1000
                    is_realtime = avg_time_ms < realtime_threshold
                    status = "✓ Real-time capable" if is_realtime else "✗ Not real-time"
                    f.write(f"{tech_name}: {avg_time_ms:.2f}ms - {status}\n")
            
            f.write("\nEDGE DEVICE RECOMMENDATIONS\n")
            f.write("-" * 30 + "\n")
            
            # Sort techniques by performance
            sorted_techs = sorted(
                [(name, result) for name, result in self.results['technique_benchmarks'].items() 
                 if 'error' not in result],
                key=lambda x: x[1]['mean_time']
            )
            
            f.write("Recommended order for edge deployment:\n")
            for i, (tech_name, result) in enumerate(sorted_techs, 1):
                avg_time_ms = result['mean_time'] * 1000
                memory_mb = result['mean_memory_mb']
                f.write(f"{i}. {tech_name}: {avg_time_ms:.2f}ms, {memory_mb:.1f}MB\n")
        
        print(f"Report saved to {output_file}")
        print(f"Summary saved to {summary_file}")

def main():
    """Main execution function"""
    
    # Check for required files
    model_file = 'best_yolo11.pt'
    test_video = 'test_video.mp4'
    
    if not os.path.exists(model_file):
        print(f"Warning: Model file {model_file} not found. Using YOLOv8 as fallback.")
        model_file = 'yolov8n.pt'
    
    # Initialize benchmark
    benchmark = MaskingPerformanceBenchmark(model_file, test_video)
    
    try:
        # Run comprehensive benchmark
        results = benchmark.run_comprehensive_benchmark()
        
        # Generate reports
        benchmark.generate_report('masking_performance_report.json')
        
        print("\n" + "=" * 60)
        print("BENCHMARK COMPLETED SUCCESSFULLY")
        print("=" * 60)
        
        # Quick summary
        if results['technique_benchmarks']:
            fastest = min(results['technique_benchmarks'].items(), 
                         key=lambda x: x[1]['mean_time'] if 'error' not in x[1] else float('inf'))
            slowest = max(results['technique_benchmarks'].items(), 
                         key=lambda x: x[1]['mean_time'] if 'error' not in x[1] else 0)
            
            print(f"Fastest technique: {fastest[0]} ({fastest[1]['mean_time']*1000:.2f}ms)")
            print(f"Slowest technique: {slowest[0]} ({slowest[1]['mean_time']*1000:.2f}ms)")
            
            # Real-time analysis
            realtime_capable = [name for name, result in results['technique_benchmarks'].items() 
                              if 'error' not in result and result['mean_time'] * 1000 < 33.33]
            
            print(f"Real-time capable techniques: {len(realtime_capable)}/{len(results['technique_benchmarks'])}")
            
    except Exception as e:
        print(f"Benchmark failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
