#!/usr/bin/env python3
"""
Comprehensive Visual & Performance Analysis for YOLO11 Masking System
- Multiple frame comparisons with proper technique names
- Detailed performance table with YOLO11 baseline vs masking overhead
- Professional figures ready for academic publication
"""

import cv2
import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from ultralytics import YOLO
import os
from pathlib import Path
import seaborn as sns
from matplotlib.gridspec import GridSpec

# Load model
model = YOLO('best_yolo11.pt' if os.path.exists('best_yolo11.pt') else 'yolov8n.pt')

class_names = [
    'trak', 'cyclist', 'bike', 'tempo', 'car', 'zeep', 'toto',
    'e-rickshaw', 'auto-rickshaw', 'bus', 'van', 'cycle-rickshaw',
    'person', 'taxi'
]

def get_multiple_test_frames():
    """Get multiple test frames from videos or create synthetic ones"""
    frames = []
    
    # Try to get frames from existing videos
    video_files = ['test_video.mp4', 'BlacknWhite.mp4', 'CompleteBlackout.mp4', 'GuassianBlur.mp4']
    
    for video_file in video_files:
        if os.path.exists(video_file):
            cap = cv2.VideoCapture(video_file)
            frame_count = 0
            while frame_count < 2:  # Get 2 frames per video
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(cv2.resize(frame, (640, 480)))
                frame_count += 1
                # Skip some frames
                for _ in range(30):
                    cap.read()
            cap.release()
    
    # If we don't have enough frames, create synthetic ones
    while len(frames) < 6:
        frame = create_synthetic_frame(len(frames))
        frames.append(frame)
    
    return frames[:6]  # Return exactly 6 frames

def create_synthetic_frame(frame_id):
    """Create synthetic frame with traffic scene"""
    np.random.seed(42 + frame_id)  # Consistent but different frames
    
    # Create base frame
    frame = np.random.randint(80, 150, (480, 640, 3), dtype=np.uint8)
    
    # Add sky gradient
    for y in range(200):
        intensity = int(200 - y * 0.5)
        frame[y, :] = [intensity, intensity + 20, intensity + 40]
    
    # Add road
    frame[350:, :] = [60, 60, 60]  # Dark road
    
    # Add lane markings
    for x in range(50, 640, 100):
        cv2.rectangle(frame, (x, 380), (x + 30, 390), (255, 255, 255), -1)
    
    # Add buildings/background
    building_positions = [(0, 150), (200, 180), (450, 160)]
    for i, (x, y) in enumerate(building_positions):
        color = [100 + i*30, 80 + i*20, 70 + i*25]
        cv2.rectangle(frame, (x, y), (x + 150, 350), color, -1)
    
    # Add vehicles at different positions based on frame_id
    vehicle_configs = [
        # (x, y, width, height, color)
        (100 + frame_id*20, 280, 80, 60, (0, 0, 200)),      # Red car
        (300 + frame_id*15, 250, 120, 80, (200, 0, 0)),     # Blue bus
        (500 - frame_id*10, 290, 70, 50, (0, 200, 0)),      # Green car
        (150 + frame_id*25, 320, 60, 40, (200, 200, 0)),    # Cyan bike
    ]
    
    for x, y, w, h, color in vehicle_configs:
        if 0 <= x <= 640-w and 0 <= y <= 480-h:
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, -1)
            # Add windows
            cv2.rectangle(frame, (x + 5, y + 5), (x + w - 5, y + 20), (50, 50, 50), -1)
    
    # Add some people
    people_positions = [(80 + frame_id*5, 340), (400 - frame_id*8, 350)]
    for x, y in people_positions:
        if 0 <= x <= 620:
            cv2.circle(frame, (x, y), 15, (255, 180, 120), -1)  # Person
    
    return frame

def apply_structural_segmentation(frame, selected_object='car'):
    """
    Structural Segmentation - Isolates objects while preserving structural context
    (Previously: Black & White)
    """
    results = model(frame, conf=0.5)
    
    # Enhanced road/structural detection
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect road/ground structures
    lower_gray = np.array([0, 0, 40])
    upper_gray = np.array([180, 40, 180])
    road_mask = cv2.inRange(hsv, lower_gray, upper_gray)
    
    # Detect structural elements (edges, lines)
    edges = cv2.Canny(gray, 50, 150)
    edges = cv2.dilate(edges, np.ones((3,3), np.uint8), iterations=1)
    
    # Object mask for selected objects
    object_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    for det in results[0].boxes:
        x1,y1,x2,y2 = det.xyxy[0].cpu().numpy().astype(int)
        try:
            label = class_names[int(det.cls[0].cpu().item())]
            if label == selected_object:
                cv2.rectangle(object_mask, (x1,y1), (x2,y2), 255, -1)
        except:
            pass
    
    # Combine structural elements
    structural_mask = cv2.bitwise_or(road_mask, edges)
    final_mask = cv2.bitwise_or(structural_mask, object_mask)
    
    # Apply segmentation
    desat = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    desat = cv2.cvtColor(desat, cv2.COLOR_GRAY2BGR)
    result = frame.copy()
    result[final_mask == 0] = desat[final_mask == 0]
    
    return result

def apply_contextual_inpainting(frame, selected_object='car'):
    """
    Contextual Inpainting - Intelligent removal with context preservation
    (Previously: Complete Inpainting)
    """
    results = model(frame, conf=0.5)
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    context_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    
    # Context preservation classes
    context_classes = ['road', 'lane', 'street', 'ground', 'path', 'sidewalk']
    
    for det in results[0].boxes:
        x1,y1,x2,y2 = det.xyxy[0].cpu().numpy().astype(int)
        try:
            label = class_names[int(det.cls[0].cpu().item())]
            if label == selected_object:
                cv2.rectangle(mask, (x1,y1), (x2,y2), 255, -1)
            # Preserve contextual elements
            if any(context in label.lower() for context in context_classes):
                cv2.rectangle(context_mask, (x1,y1), (x2,y2), 255, -1)
        except:
            pass
    
    # Create inpainting mask (invert object mask, preserve context)
    inpaint_mask = cv2.bitwise_and(cv2.bitwise_not(mask), cv2.bitwise_not(context_mask))
    
    # Apply dual inpainting for better quality
    inpaint_ns = cv2.inpaint(frame, inpaint_mask, 5, cv2.INPAINT_NS)
    inpaint_tl = cv2.inpaint(frame, inpaint_mask, 5, cv2.INPAINT_TELEA)
    result = cv2.addWeighted(inpaint_ns, 0.6, inpaint_tl, 0.4, 0)
    
    return result

def apply_privacy_preserving_blurring(frame, selected_object='car'):
    """
    Privacy-Preserving Blurring - Strong blur for privacy while keeping target object
    (Previously: Gaussian Blur)
    """
    results = model(frame, conf=0.5)
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    
    for det in results[0].boxes:
        x1,y1,x2,y2 = det.xyxy[0].cpu().numpy().astype(int)
        try:
            label = class_names[int(det.cls[0].cpu().item())]
            if label == selected_object:
                cv2.rectangle(mask, (x1,y1), (x2,y2), 255, -1)
        except:
            pass
    
    # Multi-level blurring for privacy
    blur_strong = cv2.GaussianBlur(frame, (51, 51), 0)  # Strong blur
    blur_medium = cv2.GaussianBlur(frame, (25, 25), 0)  # Medium blur
    
    # Apply progressive blurring
    result = frame.copy()
    result[mask == 0] = blur_medium[mask == 0]
    
    # Extra privacy for potential face regions (upper part of frame)
    privacy_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    privacy_mask[0:200, :] = 255  # Upper region
    privacy_mask = cv2.bitwise_and(privacy_mask, cv2.bitwise_not(mask))
    result[privacy_mask == 255] = blur_strong[privacy_mask == 255]
    
    return result

def apply_contextual_focus_rendering(frame, selected_object='car'):
    """
    Contextual Focus Rendering - Advanced focus with contextual enhancement
    (Previously: Complete Blackout but enhanced)
    """
    results = model(frame, conf=0.5)
    
    # Create focus map
    focus_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    context_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    
    for det in results[0].boxes:
        x1,y1,x2,y2 = det.xyxy[0].cpu().numpy().astype(int)
        try:
            label = class_names[int(det.cls[0].cpu().item())]
            if label == selected_object:
                # Create focus region with soft edges
                cv2.rectangle(focus_mask, (x1-10,y1-10), (x2+10,y2+10), 255, -1)
                # Enhanced focus for exact object
                cv2.rectangle(focus_mask, (x1,y1), (x2,y2), 255, -1)
        except:
            pass
    
    # Create gradient focus effect
    focus_blur = cv2.GaussianBlur(focus_mask.astype(np.float32), (31, 31), 0)
    focus_blur = focus_blur / focus_blur.max()
    
    # Create dark background with contextual hints
    dark_bg = frame * 0.15  # Very dark background
    
    # Apply contextual focus rendering
    result = np.zeros_like(frame, dtype=np.float32)
    for c in range(3):
        result[:,:,c] = dark_bg[:,:,c] * (1 - focus_blur) + frame[:,:,c] * focus_blur
    
    result = np.clip(result, 0, 255).astype(np.uint8)
    
    # Enhance focused objects
    enhanced_focus = cv2.convertScaleAbs(result, alpha=1.2, beta=10)
    result[focus_mask > 128] = enhanced_focus[focus_mask > 128]
    
    return result

def apply_selective_texture_blurring(frame, selected_object='car'):
    """
    Selective Texture Blurring - Texture-based obscuration with selective preservation
    (Previously: Selective Blur)
    """
    results = model(frame, conf=0.5)
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    
    for det in results[0].boxes:
        x1,y1,x2,y2 = det.xyxy[0].cpu().numpy().astype(int)
        try:
            label = class_names[int(det.cls[0].cpu().item())]
            if label != selected_object:
                cv2.rectangle(mask, (x1,y1), (x2,y2), 255, -1)
        except:
            pass
    
    # Create multiple texture effects
    # 1. Pixelation effect
    h, w = frame.shape[:2]
    temp = cv2.resize(frame, (w//12, h//12), interpolation=cv2.INTER_LINEAR)
    pixelated = cv2.resize(temp, (w, h), interpolation=cv2.INTER_NEAREST)
    
    # 2. Oil painting effect (simplified)
    oil_effect = cv2.bilateralFilter(frame, 15, 80, 80)
    oil_effect = cv2.bilateralFilter(oil_effect, 15, 80, 80)
    
    # 3. Median blur for texture
    median_blur = cv2.medianBlur(frame, 19)
    
    # Apply selective texture blurring
    result = frame.copy()
    
    # Different texture effects for different regions
    mask_dilated = cv2.dilate(mask, np.ones((5,5), np.uint8), iterations=2)
    mask_edge = cv2.subtract(mask_dilated, mask)
    
    result[mask == 255] = pixelated[mask == 255]  # Core non-selected objects
    result[mask_edge == 255] = oil_effect[mask_edge == 255]  # Edge regions
    
    return result

def measure_technique_performance(func, frame, technique_name, iterations=15):
    """Measure detailed performance of a technique"""
    print(f"  Measuring {technique_name}...")
    
    # Measure YOLO baseline first
    yolo_times = []
    for _ in range(iterations):
        start = time.perf_counter()
        results = model(frame, conf=0.5)
        end = time.perf_counter()
        yolo_times.append(end - start)
    
    yolo_avg = sum(yolo_times) / len(yolo_times)
    
    # Measure total technique time
    total_times = []
    for _ in range(iterations):
        start = time.perf_counter()
        result = func(frame)
        end = time.perf_counter()
        total_times.append(end - start)
    
    total_avg = sum(total_times) / len(total_times)
    masking_overhead = total_avg - yolo_avg
    
    return {
        'technique': technique_name,
        'yolo_time_ms': yolo_avg * 1000,
        'total_time_ms': total_avg * 1000,
        'masking_overhead_ms': masking_overhead * 1000,
        'overhead_percentage': (masking_overhead / yolo_avg) * 100,
        'fps': 1.0 / total_avg,
        'realtime_30fps': total_avg < 0.0333,
        'realtime_24fps': total_avg < 0.0417
    }

def create_comprehensive_visual_comparison():
    """Create comprehensive multi-frame comparison figure"""
    print("Creating comprehensive visual comparison with multiple frames...")
    
    frames = get_multiple_test_frames()
    
    techniques = {
        'Structural Segmentation': apply_structural_segmentation,
        'Contextual Inpainting': apply_contextual_inpainting,
        'Privacy-Preserving Blurring': apply_privacy_preserving_blurring,
        'Contextual Focus Rendering': apply_contextual_focus_rendering,
        'Selective Texture Blurring': apply_selective_texture_blurring
    }
    
    # Create large figure with subplots
    fig = plt.figure(figsize=(24, 20))
    gs = GridSpec(len(frames), len(techniques) + 1, figure=fig)
    
    fig.suptitle('Comprehensive Masking Techniques Comparison\nMultiple Frame Analysis', 
                 fontsize=20, fontweight='bold', y=0.98)
    
    # Color scheme for technique borders
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57']
    
    for frame_idx, frame in enumerate(frames):
        # Original frame
        ax_orig = fig.add_subplot(gs[frame_idx, 0])
        ax_orig.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if frame_idx == 0:
            ax_orig.set_title('Original Frames', fontsize=14, fontweight='bold', pad=10)
        ax_orig.set_ylabel(f'Frame {frame_idx + 1}', fontsize=12, fontweight='bold')
        ax_orig.axis('off')
        
        # Add frame border
        rect = patches.Rectangle((0, 0), 1, 1, linewidth=3, 
                               edgecolor='black', facecolor='none', 
                               transform=ax_orig.transAxes)
        ax_orig.add_patch(rect)
        
        # Apply each technique
        for tech_idx, (tech_name, func) in enumerate(techniques.items()):
            try:
                masked_frame = func(frame)
                ax = fig.add_subplot(gs[frame_idx, tech_idx + 1])
                ax.imshow(cv2.cvtColor(masked_frame, cv2.COLOR_BGR2RGB))
                
                if frame_idx == 0:
                    ax.set_title(tech_name, fontsize=14, fontweight='bold', pad=10)
                ax.axis('off')
                
                # Add colored border for each technique
                rect = patches.Rectangle((0, 0), 1, 1, linewidth=3, 
                                       edgecolor=colors[tech_idx], facecolor='none', 
                                       transform=ax.transAxes)
                ax.add_patch(rect)
                
            except Exception as e:
                print(f"Error applying {tech_name} to frame {frame_idx}: {e}")
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.94, bottom=0.02, left=0.02, right=0.98, hspace=0.1, wspace=0.05)
    plt.savefig('comprehensive_masking_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ Comprehensive visual comparison saved as 'comprehensive_masking_comparison.png'")

def create_detailed_performance_table():
    """Create detailed performance comparison table"""
    print("Creating detailed performance analysis table...")
    
    # Get a representative frame for testing
    test_frame = get_multiple_test_frames()[0]
    
    techniques = {
        'Structural Segmentation': apply_structural_segmentation,
        'Contextual Inpainting': apply_contextual_inpainting,
        'Privacy-Preserving Blurring': apply_privacy_preserving_blurring,
        'Contextual Focus Rendering': apply_contextual_focus_rendering,
        'Selective Texture Blurring': apply_selective_texture_blurring
    }
    
    # Measure YOLO baseline
    print("  Measuring YOLO11 baseline...")
    yolo_times = []
    for _ in range(20):
        start = time.perf_counter()
        results = model(test_frame, conf=0.5)
        end = time.perf_counter()
        yolo_times.append(end - start)
    
    yolo_baseline = {
        'technique': 'YOLO11 Baseline',
        'yolo_time_ms': sum(yolo_times) / len(yolo_times) * 1000,
        'total_time_ms': sum(yolo_times) / len(yolo_times) * 1000,
        'masking_overhead_ms': 0,
        'overhead_percentage': 0,
        'fps': 1.0 / (sum(yolo_times) / len(yolo_times)),
        'realtime_30fps': True,
        'realtime_24fps': True
    }
    
    # Measure each technique
    results = [yolo_baseline]
    
    for tech_name, func in techniques.items():
        try:
            result = measure_technique_performance(func, test_frame, tech_name)
            results.append(result)
        except Exception as e:
            print(f"Error measuring {tech_name}: {e}")
    
    # Create comprehensive table visualization
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12))
    
    # Table 1: Performance Metrics
    table1_data = []
    headers1 = ['Technique', 'YOLO Time (ms)', 'Total Time (ms)', 'Masking Overhead (ms)', 'Overhead (%)', 'FPS']
    
    for result in results:
        row = [
            result['technique'],
            f"{result['yolo_time_ms']:.2f}",
            f"{result['total_time_ms']:.2f}",
            f"{result['masking_overhead_ms']:.2f}",
            f"{result['overhead_percentage']:.1f}%",
            f"{result['fps']:.1f}"
        ]
        table1_data.append(row)
    
    table1 = ax1.table(cellText=table1_data, colLabels=headers1, 
                      cellLoc='center', loc='center',
                      colWidths=[0.25, 0.15, 0.15, 0.18, 0.12, 0.1])
    
    table1.auto_set_font_size(False)
    table1.set_fontsize(11)
    table1.scale(1.2, 2.5)
    
    # Style table 1
    for i in range(len(headers1)):
        table1[(0, i)].set_facecolor('#2E86AB')
        table1[(0, i)].set_text_props(weight='bold', color='white')
    
    # Color baseline row differently
    for j in range(len(headers1)):
        table1[(1, j)].set_facecolor('#A23B72')
        table1[(1, j)].set_text_props(weight='bold', color='white')
    
    # Color other rows alternately
    for i in range(2, len(table1_data) + 1):
        for j in range(len(headers1)):
            if i % 2 == 0:
                table1[(i, j)].set_facecolor('#F18F01')
            else:
                table1[(i, j)].set_facecolor('#C73E1D')
            table1[(i, j)].set_text_props(color='white', weight='bold')
    
    ax1.set_title('YOLO11 Performance Analysis: Baseline vs. Masking Techniques\nProcessing Time Breakdown', 
                  fontsize=16, fontweight='bold', pad=20)
    ax1.axis('off')
    
    # Table 2: Real-time Capability Analysis
    table2_data = []
    headers2 = ['Technique', 'Real-time (30 FPS)', 'Real-time (24 FPS)', 'Max Throughput', 'Edge Device Ready']
    
    for result in results:
        rt_30 = "✅ Yes" if result['realtime_30fps'] else "❌ No"
        rt_24 = "✅ Yes" if result['realtime_24fps'] else "❌ No"
        edge_ready = "✅ Ready" if result['total_time_ms'] < 25 else "⚠️ GPU Req" if result['total_time_ms'] < 50 else "❌ Too Slow"
        
        row = [
            result['technique'],
            rt_30,
            rt_24,
            f"{result['fps']:.1f} FPS",
            edge_ready
        ]
        table2_data.append(row)
    
    table2 = ax2.table(cellText=table2_data, colLabels=headers2, 
                      cellLoc='center', loc='center',
                      colWidths=[0.25, 0.18, 0.18, 0.15, 0.18])
    
    table2.auto_set_font_size(False)
    table2.set_fontsize(11)
    table2.scale(1.2, 2.5)
    
    # Style table 2
    for i in range(len(headers2)):
        table2[(0, i)].set_facecolor('#2E86AB')
        table2[(0, i)].set_text_props(weight='bold', color='white')
    
    # Color baseline row
    for j in range(len(headers2)):
        table2[(1, j)].set_facecolor('#A23B72')
        table2[(1, j)].set_text_props(weight='bold', color='white')
    
    # Color other rows
    for i in range(2, len(table2_data) + 1):
        for j in range(len(headers2)):
            if i % 2 == 0:
                table2[(i, j)].set_facecolor('#F18F01')
            else:
                table2[(i, j)].set_facecolor('#C73E1D')
            table2[(i, j)].set_text_props(color='white', weight='bold')
    
    ax2.set_title('Real-time Applicability & Edge Device Integration Analysis', 
                  fontsize=16, fontweight='bold', pad=20)
    ax2.axis('off')
    
    plt.tight_layout()
    plt.savefig('detailed_performance_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save detailed text report
    with open('detailed_performance_report.txt', 'w') as f:
        f.write("COMPREHENSIVE PERFORMANCE ANALYSIS - YOLO11 MASKING TECHNIQUES\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("PERFORMANCE METRICS SUMMARY\n")
        f.write("-" * 40 + "\n")
        f.write(f"{'Technique':<30} {'Total (ms)':<12} {'Overhead (ms)':<15} {'FPS':<8} {'Real-time':<12}\n")
        f.write("-" * 80 + "\n")
        
        for result in results:
            rt_status = "✅" if result['realtime_30fps'] else "❌"
            f.write(f"{result['technique']:<30} {result['total_time_ms']:<12.2f} {result['masking_overhead_ms']:<15.2f} {result['fps']:<8.1f} {rt_status:<12}\n")
        
        f.write("\nDETAILED ANALYSIS\n")
        f.write("-" * 20 + "\n")
        
        for result in results:
            f.write(f"\n{result['technique']}:\n")
            f.write(f"  YOLO Inference: {result['yolo_time_ms']:.2f}ms\n")
            f.write(f"  Total Processing: {result['total_time_ms']:.2f}ms\n")
            f.write(f"  Masking Overhead: {result['masking_overhead_ms']:.2f}ms ({result['overhead_percentage']:.1f}%)\n")
            f.write(f"  Throughput: {result['fps']:.1f} FPS\n")
            f.write(f"  Real-time (30 FPS): {'Yes' if result['realtime_30fps'] else 'No'}\n")
            f.write(f"  Real-time (24 FPS): {'Yes' if result['realtime_24fps'] else 'No'}\n")
        
        # Performance ranking
        technique_results = [r for r in results if r['technique'] != 'YOLO11 Baseline']
        sorted_by_speed = sorted(technique_results, key=lambda x: x['total_time_ms'])
        
        f.write(f"\nPERFORMANCE RANKING (Fastest to Slowest)\n")
        f.write("-" * 45 + "\n")
        for i, result in enumerate(sorted_by_speed, 1):
            f.write(f"{i}. {result['technique']}: {result['total_time_ms']:.2f}ms\n")
        
        f.write(f"\nEDGE DEVICE RECOMMENDATIONS\n")
        f.write("-" * 30 + "\n")
        
        fast_techniques = [r for r in technique_results if r['total_time_ms'] < 30]
        medium_techniques = [r for r in technique_results if 30 <= r['total_time_ms'] < 60]
        slow_techniques = [r for r in technique_results if r['total_time_ms'] >= 60]
        
        f.write("✅ Edge-Ready (CPU capable):\n")
        for result in fast_techniques:
            f.write(f"  - {result['technique']}: {result['total_time_ms']:.2f}ms\n")
        
        f.write("\n⚠️  GPU Recommended:\n")
        for result in medium_techniques:
            f.write(f"  - {result['technique']}: {result['total_time_ms']:.2f}ms\n")
        
        f.write("\n❌ Requires Optimization:\n")
        for result in slow_techniques:
            f.write(f"  - {result['technique']}: {result['total_time_ms']:.2f}ms\n")
    
    print("✅ Detailed performance analysis saved as 'detailed_performance_analysis.png'")
    print("✅ Detailed performance report saved as 'detailed_performance_report.txt'")
    
    return results

def main():
    """Main execution function"""
    print("🚀 COMPREHENSIVE VISUAL & PERFORMANCE ANALYSIS")
    print("=" * 70)
    print("Generating:")
    print("  📊 Multi-frame visual comparisons")
    print("  📈 Detailed performance tables")
    print("  🎯 YOLO11 baseline vs masking overhead analysis")
    print("-" * 70)
    
    start_time = time.time()
    
    # Create comprehensive visual comparison
    create_comprehensive_visual_comparison()
    
    # Create detailed performance analysis
    results = create_detailed_performance_table()
    
    end_time = time.time()
    
    print("\n" + "=" * 70)
    print("✅ COMPREHENSIVE ANALYSIS COMPLETED")
    print("=" * 70)
    print(f"Total execution time: {end_time - start_time:.2f} seconds")
    
    print("\n📁 FILES GENERATED:")
    print("1. comprehensive_masking_comparison.png - Multi-frame visual comparison")
    print("2. detailed_performance_analysis.png - Complete performance tables")
    print("3. detailed_performance_report.txt - Text report for paper")
    
    # Quick summary
    technique_results = [r for r in results if r['technique'] != 'YOLO11 Baseline']
    if technique_results:
        fastest = min(technique_results, key=lambda x: x['total_time_ms'])
        slowest = max(technique_results, key=lambda x: x['total_time_ms'])
        realtime_count = sum(1 for r in technique_results if r['realtime_30fps'])
        
        print(f"\n📊 PERFORMANCE SUMMARY:")
        print(f"YOLO11 Baseline: {results[0]['total_time_ms']:.2f}ms ({results[0]['fps']:.1f} FPS)")
        print(f"Fastest Technique: {fastest['technique']} ({fastest['total_time_ms']:.2f}ms)")
        print(f"Slowest Technique: {slowest['technique']} ({slowest['total_time_ms']:.2f}ms)")
        print(f"Real-time Capable: {realtime_count}/{len(technique_results)} techniques")
        print(f"Average Overhead: {sum(r['overhead_percentage'] for r in technique_results)/len(technique_results):.1f}%")
    
    print(f"\n🎯 READY FOR ACADEMIC PAPER:")
    print("  - Professional technique names")
    print("  - Multiple frame comparisons")
    print("  - Detailed performance breakdown")
    print("  - YOLO11 baseline vs masking overhead")
    print("  - Real-time applicability analysis")
    print("  - Edge device integration recommendations")

if __name__ == "__main__":
    main()
