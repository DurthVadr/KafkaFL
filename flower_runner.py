#!/usr/bin/env python3
"""
Direct Flower implementation runner with result plotting
"""

import subprocess
import time
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime
import re
import threading
import queue

def start_flower_server():
    """Start Flower server and capture output"""
    print("📡 Starting Flower server...")
    
    os.chdir("flower_implementation")
    
    server_process = subprocess.Popen([
        sys.executable, "server_app.py"
    ], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, 
    text=True, bufsize=1, universal_newlines=True)
    
    return server_process

def start_flower_client(client_id):
    """Start a single Flower client"""
    print(f"🤖 Starting client {client_id}...")
    
    client_process = subprocess.Popen([
        sys.executable, "client_app.py", 
        "--client_id", str(client_id),
        "--num_clients", "3"
    ], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, 
    text=True, bufsize=1, universal_newlines=True)
    
    return client_process

def monitor_process_output(process, output_queue, process_name):
    """Monitor process output in a separate thread"""
    try:
        for line in iter(process.stdout.readline, ''):
            if line:
                output_queue.put((process_name, line.strip()))
        process.stdout.close()
    except:
        pass

def run_flower_federated_learning():
    """Run complete Flower federated learning with monitoring"""
    
    print("🌸 Starting Flower Federated Learning")
    print("=" * 60)
    
    original_dir = os.getcwd()
    all_output = []
    processes = []
    
    try:
        # Start server
        server_process = start_flower_server()
        processes.append(("server", server_process))
        
        # Wait for server to initialize
        print("⏳ Waiting for server to initialize...")
        time.sleep(8)
        
        # Check if server started successfully
        if server_process.poll() is not None:
            print("❌ Server failed to start!")
            stdout, _ = server_process.communicate()
            print("Server output:", stdout)
            return None
        
        print("✅ Server started successfully")
        
        # Start clients with delays
        for client_id in range(3):
            client_process = start_flower_client(client_id)
            processes.append((f"client_{client_id}", client_process))
            time.sleep(3)  # Delay between clients
        
        print("✅ All clients started")
        
        # Monitor all processes
        output_queue = queue.Queue()
        monitor_threads = []
        
        for process_name, process in processes:
            thread = threading.Thread(
                target=monitor_process_output, 
                args=(process, output_queue, process_name)
            )
            thread.daemon = True
            thread.start()
            monitor_threads.append(thread)
        
        print("\n📊 Monitoring federated learning progress...")
        print("⏹️  Press Ctrl+C to stop\n")
        
        # Collect output with timeout
        start_time = time.time()
        timeout = 600  # 10 minutes
        
        while True:
            try:
                # Check for output with timeout
                try:
                    process_name, line = output_queue.get(timeout=1)
                    all_output.append(line)
                    
                    # Print important lines
                    if any(keyword in line for keyword in [
                        "Server eval round", "accuracy", "loss", "Round", "Client"
                    ]):
                        print(f"[{process_name}] {line}")
                        
                except queue.Empty:
                    pass
                
                # Check if all processes are done
                running_processes = [p for _, p in processes if p.poll() is None]
                if not running_processes:
                    print("\n✅ All processes completed!")
                    break
                
                # Check timeout
                if time.time() - start_time > timeout:
                    print(f"\n⏰ Timeout reached ({timeout}s)")
                    break
                    
            except KeyboardInterrupt:
                print("\n⏹️  Stopping federated learning...")
                break
        
        # Collect remaining output
        time.sleep(2)
        while not output_queue.empty():
            try:
                _, line = output_queue.get_nowait()
                all_output.append(line)
            except queue.Empty:
                break
        
        return all_output
        
    finally:
        # Clean up processes
        for process_name, process in processes:
            try:
                if process.poll() is None:
                    process.terminate()
                    process.wait(timeout=5)
                    print(f"🔴 {process_name} stopped")
            except subprocess.TimeoutExpired:
                try:
                    process.kill()
                    print(f"🔴 {process_name} force killed")
                except:
                    pass
            except:
                pass
        
        os.chdir(original_dir)

def parse_flower_output(output_lines):
    """Parse Flower output to extract metrics"""
    
    server_rounds = []
    server_accuracies = []
    server_losses = []
    
    print(f"\n🔍 Parsing {len(output_lines)} output lines...")
    
    for line in output_lines:
        # Look for server evaluation results
        if "Server eval round" in line:
            try:
                # Parse: "Server eval round 1: loss=0.1234, accuracy=0.5678"
                round_match = re.search(r'round (\d+)', line)
                loss_match = re.search(r'loss=([0-9.]+)', line)
                acc_match = re.search(r'accuracy=([0-9.]+)', line)
                
                if round_match and loss_match and acc_match:
                    round_num = int(round_match.group(1))
                    loss = float(loss_match.group(1))
                    accuracy = float(acc_match.group(1))
                    
                    server_rounds.append(round_num)
                    server_losses.append(loss)
                    server_accuracies.append(accuracy)
                    
                    print(f"  📈 Parsed Round {round_num}: Loss={loss:.4f}, Accuracy={accuracy:.4f}")
                    
            except Exception as e:
                print(f"  ⚠️  Failed to parse: {line[:50]}... ({e})")
    
    print(f"📊 Successfully parsed {len(server_rounds)} evaluation rounds")
    
    return {
        'rounds': server_rounds,
        'accuracies': server_accuracies,
        'losses': server_losses,
        'raw_output': output_lines
    }

def create_flower_plots(results):
    """Create comprehensive plots for Flower results"""
    
    if not results or not results['accuracies']:
        print("❌ No valid results to plot")
        return
    
    rounds = results['rounds']
    accuracies = results['accuracies']
    losses = results['losses']
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Flower Federated Learning Results', fontsize=16, fontweight='bold')
    
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
    
    # Plot 1: Accuracy over rounds
    axes[0, 0].plot(rounds, accuracies, 'o-', color=colors[0], linewidth=2, markersize=6)
    axes[0, 0].set_xlabel('Communication Round')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].set_title('Server-side Accuracy')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_ylim(0, 1)
    
    # Plot 2: Loss over rounds
    axes[0, 1].plot(rounds, losses, 's-', color=colors[1], linewidth=2, markersize=6)
    axes[0, 1].set_xlabel('Communication Round')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].set_title('Server-side Loss')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Accuracy improvement per round
    if len(accuracies) > 1:
        improvements = np.diff(accuracies)
        improvement_rounds = rounds[1:]
        colors_bar = ['green' if x > 0 else 'red' for x in improvements]
        axes[0, 2].bar(improvement_rounds, improvements, color=colors_bar, alpha=0.7)
        axes[0, 2].set_xlabel('Communication Round')
        axes[0, 2].set_ylabel('Accuracy Improvement')
        axes[0, 2].set_title('Round-to-Round Improvement')
        axes[0, 2].grid(True, alpha=0.3)
        axes[0, 2].axhline(y=0, color='black', linestyle='-', alpha=0.5)
    
    # Plot 4: Performance statistics
    if accuracies:
        stats = {
            'Initial': accuracies[0],
            'Final': accuracies[-1],
            'Best': max(accuracies),
            'Average': np.mean(accuracies)
        }
        
        bars = axes[1, 0].bar(stats.keys(), stats.values(), color=colors)
        axes[1, 0].set_ylabel('Accuracy')
        axes[1, 0].set_title('Performance Statistics')
        axes[1, 0].set_ylim(0, 1)
        
        # Add value labels
        for bar, value in zip(bars, stats.values()):
            height = bar.get_height()
            axes[1, 0].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{value:.3f}', ha='center', va='bottom')
    
    # Plot 5: Convergence analysis (moving average)
    if len(accuracies) > 3:
        window_size = min(3, len(accuracies) // 2)
        moving_avg = np.convolve(accuracies, np.ones(window_size)/window_size, mode='valid')
        moving_rounds = rounds[window_size-1:]
        
        axes[1, 1].plot(rounds, accuracies, 'o-', alpha=0.5, color=colors[0], label='Raw')
        axes[1, 1].plot(moving_rounds, moving_avg, '-', linewidth=3, color=colors[1], 
                       label=f'{window_size}-Round Avg')
        axes[1, 1].set_xlabel('Communication Round')
        axes[1, 1].set_ylabel('Accuracy')
        axes[1, 1].set_title('Convergence Analysis')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].legend()
    
    # Plot 6: Learning curve (both metrics)
    ax6 = axes[1, 2]
    ax6_twin = ax6.twinx()
    
    line1 = ax6.plot(rounds, accuracies, 'o-', color=colors[0], label='Accuracy')
    line2 = ax6_twin.plot(rounds, losses, 's-', color=colors[1], label='Loss')
    
    ax6.set_xlabel('Communication Round')
    ax6.set_ylabel('Accuracy', color=colors[0])
    ax6_twin.set_ylabel('Loss', color=colors[1])
    ax6.set_title('Learning Curve')
    ax6.grid(True, alpha=0.3)
    
    # Combined legend
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax6.legend(lines, labels, loc='center right')
    
    plt.tight_layout()
    
    # Save plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"flower_results_{timestamp}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"📊 Plot saved as: {filename}")
    
    plt.show()

def save_detailed_results(results):
    """Save detailed results to files"""
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save CSV with metrics
    if results['accuracies']:
        df = pd.DataFrame({
            'Round': results['rounds'],
            'Accuracy': results['accuracies'],
            'Loss': results['losses']
        })
        csv_file = f"flower_metrics_{timestamp}.csv"
        df.to_csv(csv_file, index=False)
        print(f"💾 Metrics saved as: {csv_file}")
    
    # Save raw output
    output_file = f"flower_output_{timestamp}.txt"
    with open(output_file, 'w') as f:
        f.write("Flower Federated Learning Output\n")
        f.write("=" * 50 + "\n")
        f.write(f"Generated: {datetime.now()}\n\n")
        for line in results['raw_output']:
            f.write(line + "\n")
    print(f"💾 Raw output saved as: {output_file}")

def main():
    """Main function to run Flower and create plots"""
    
    print("🌸 Flower Federated Learning Direct Runner")
    print("=" * 60)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Run Flower federated learning
    output = run_flower_federated_learning()
    
    if output:
        print(f"\n📊 Processing {len(output)} output lines...")
        
        # Parse results
        results = parse_flower_output(output)
        
        if results['accuracies']:
            # Create plots
            print("\n📈 Creating performance plots...")
            create_flower_plots(results)
            
            # Save results
            save_detailed_results(results)
            
            # Print summary
            print(f"\n📋 Summary:")
            print(f"  Rounds completed: {len(results['rounds'])}")
            print(f"  Final accuracy: {results['accuracies'][-1]:.4f}")
            print(f"  Best accuracy: {max(results['accuracies']):.4f}")
            print(f"  Total improvement: {results['accuracies'][-1] - results['accuracies'][0]:.4f}")
            
        else:
            print("❌ No metrics found in output")
    else:
        print("❌ No output captured")
    
    print(f"\n🏁 Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()