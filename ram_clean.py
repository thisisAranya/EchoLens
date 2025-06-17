#!/usr/bin/env python3
"""
RAM Cleanup Utility
Simple script to free up system RAM and monitor memory usage
"""

import gc
import os
import sys
import time
import psutil
import subprocess
from typing import Dict, Any

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

class RAMCleaner:
    """RAM cleanup utility class"""
    
    def __init__(self):
        self.initial_memory = self.get_memory_stats()
        
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get current memory statistics"""
        memory = psutil.virtual_memory()
        process = psutil.Process()
        process_memory = process.memory_info()
        
        return {
            "total_gb": round(memory.total / (1024**3), 2),
            "used_gb": round(memory.used / (1024**3), 2),
            "available_gb": round(memory.available / (1024**3), 2),
            "percent_used": memory.percent,
            "process_mb": round(process_memory.rss / (1024**2), 1)
        }
    
    def cleanup_python_memory(self) -> int:
        """Clean up Python garbage collection"""
        print("🐍 Cleaning Python memory...")
        collected = gc.collect()
        print(f"   Collected {collected} objects")
        return collected
    
    def cleanup_torch_memory(self) -> bool:
        """Clean up PyTorch CUDA cache"""
        if not TORCH_AVAILABLE:
            print("⚠️  PyTorch not available")
            return False
            
        if torch.cuda.is_available():
            print("🔥 Cleaning PyTorch CUDA cache...")
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            
            # Get GPU memory info
            gpu_memory = torch.cuda.memory_allocated() / (1024**2)
            print(f"   GPU memory: {gpu_memory:.1f}MB")
            return True
        else:
            print("💻 CUDA not available, skipping GPU cleanup")
            return False
    
    def cleanup_system_cache(self) -> bool:
        """Clean up system cache (Windows/Linux)"""
        try:
            if os.name == 'nt':  # Windows
                print("🪟 Cleaning Windows system cache...")
                # Clear DNS cache
                subprocess.run(['ipconfig', '/flushdns'], 
                             capture_output=True, check=False)
                return True
            else:  # Linux/Mac
                print("🐧 Cleaning Linux system cache...")
                # Clear page cache (requires sudo)
                try:
                    subprocess.run(['sudo', 'sync'], 
                                 capture_output=True, check=False)
                    subprocess.run(['sudo', 'echo', '1', '>', '/proc/sys/vm/drop_caches'], 
                                 capture_output=True, check=False)
                except:
                    print("   ⚠️  Need sudo for system cache cleanup")
                return True
        except Exception as e:
            print(f"   ❌ System cleanup failed: {e}")
            return False
    
    def force_garbage_collection(self, iterations: int = 3) -> int:
        """Force multiple garbage collection cycles"""
        print(f"🗑️  Running {iterations} garbage collection cycles...")
        total_collected = 0
        
        for i in range(iterations):
            collected = gc.collect()
            total_collected += collected
            print(f"   Cycle {i+1}: {collected} objects")
            time.sleep(0.1)  # Small delay between cycles
        
        return total_collected
    
    def kill_memory_hogs(self, threshold_gb: float = 1.0) -> int:
        """Find and optionally kill memory-intensive processes"""
        print(f"🔍 Finding processes using >{threshold_gb}GB RAM...")
        
        memory_hogs = []
        for proc in psutil.process_iter(['pid', 'name', 'memory_info']):
            try:
                memory_gb = proc.info['memory_info'].rss / (1024**3)
                if memory_gb > threshold_gb:
                    memory_hogs.append({
                        'pid': proc.info['pid'],
                        'name': proc.info['name'],
                        'memory_gb': round(memory_gb, 2)
                    })
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        
        if memory_hogs:
            print("   Memory-intensive processes:")
            for proc in sorted(memory_hogs, key=lambda x: x['memory_gb'], reverse=True):
                print(f"   • {proc['name']} (PID: {proc['pid']}): {proc['memory_gb']}GB")
        else:
            print("   No memory hogs found")
        
        return len(memory_hogs)
    
    def comprehensive_cleanup(self) -> Dict[str, Any]:
        """Run comprehensive RAM cleanup"""
        print("🧹 Starting comprehensive RAM cleanup...")
        print("=" * 50)
        
        # Before stats
        before = self.get_memory_stats()
        print(f"📊 Before cleanup: {before['used_gb']}GB used ({before['percent_used']:.1f}%)")
        
        # Cleanup steps
        results = {
            "python_objects_collected": self.force_garbage_collection(),
            "torch_cleaned": self.cleanup_torch_memory(),
            "system_cleaned": False  # We'll skip system cleanup by default
        }
        
        # After stats
        after = self.get_memory_stats()
        freed_gb = before['used_gb'] - after['used_gb']
        
        results.update({
            "before_used_gb": before['used_gb'],
            "after_used_gb": after['used_gb'],
            "freed_gb": round(freed_gb, 2),
            "percent_improvement": round((freed_gb / before['used_gb']) * 100, 1) if before['used_gb'] > 0 else 0
        })
        
        print("=" * 50)
        print(f"✅ Cleanup complete!")
        print(f"📊 After cleanup: {after['used_gb']}GB used ({after['percent_used']:.1f}%)")
        print(f"🎉 Freed: {freed_gb:.2f}GB ({results['percent_improvement']:.1f}% improvement)")
        
        return results
    
    def monitor_memory(self, duration: int = 30, interval: int = 2):
        """Monitor memory usage over time"""
        print(f"📈 Monitoring memory for {duration} seconds...")
        
        start_time = time.time()
        max_usage = 0
        min_usage = float('inf')
        
        while time.time() - start_time < duration:
            stats = self.get_memory_stats()
            current_usage = stats['percent_used']
            
            max_usage = max(max_usage, current_usage)
            min_usage = min(min_usage, current_usage)
            
            print(f"RAM: {current_usage:.1f}% | Available: {stats['available_gb']}GB", end='\r')
            time.sleep(interval)
        
        print(f"\n📊 Monitor complete: Min: {min_usage:.1f}% | Max: {max_usage:.1f}%")

def main():
    """Main function with interactive menu"""
    cleaner = RAMCleaner()
    
    print("🧠 RAM Cleanup Utility")
    print("=" * 30)
    
    while True:
        print("\nOptions:")
        print("1. 📊 Show memory stats")
        print("2. 🧹 Quick cleanup")
        print("3. 🔥 Comprehensive cleanup")
        print("4. 🔍 Find memory hogs")
        print("5. 📈 Monitor memory (30s)")
        print("6. ❌ Exit")
        
        try:
            choice = input("\nChoose option (1-6): ").strip()
            
            if choice == '1':
                stats = cleaner.get_memory_stats()
                print(f"\n📊 Memory Stats:")
                print(f"   Total: {stats['total_gb']}GB")
                print(f"   Used: {stats['used_gb']}GB ({stats['percent_used']:.1f}%)")
                print(f"   Available: {stats['available_gb']}GB")
                print(f"   This process: {stats['process_mb']}MB")
                
            elif choice == '2':
                print("\n🧹 Quick cleanup...")
                collected = cleaner.cleanup_python_memory()
                cleaner.cleanup_torch_memory()
                print(f"✅ Quick cleanup done! Collected {collected} objects")
                
            elif choice == '3':
                results = cleaner.comprehensive_cleanup()
                
            elif choice == '4':
                cleaner.kill_memory_hogs()
                
            elif choice == '5':
                cleaner.monitor_memory()
                
            elif choice == '6':
                print("👋 Goodbye!")
                break
                
            else:
                print("❌ Invalid choice. Please enter 1-6.")
                
        except KeyboardInterrupt:
            print("\n👋 Interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()