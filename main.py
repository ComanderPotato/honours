import subprocess
import re
import time
from datetime import datetime
import os
import csv
def extract_memory_cumsum(file_path):
    file_path = os.path.join(file_path, 'logs.txt')
    memory_values = []
    with open(file_path, 'r') as file:
        for line in file:
            match = re.search(r"memory:\s([\d.]+)GB", line)
            if match:
                memory_values.append(float(match.group(1)))

    cumulative_sum = round(sum(memory_values), 3)
    
    average_memory = round(cumulative_sum / len(memory_values), 3) if memory_values else 0
    return memory_values, cumulative_sum, average_memory

def get_final_train_accu(file_path):
    file_path = os.path.join(file_path, 'logs.txt')

    try:
        with open(file_path, 'r') as file:
            lines = file.readlines()

        if not lines:
            return None 

        last_line = lines[-1].strip()
        while not last_line and lines:
            lines.pop()
            last_line = lines[-1].strip() if lines else None

        if not last_line:
            return None

        match = re.search(r"train/accu:\s*([\d.]+)", last_line)
        if match:
            return float(match.group(1))

        return None

    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def get_total_time(file_path):
    file_path = os.path.join(file_path, "log.csv")
    with open(file_path, 'r') as file:
        timestamps = []

        for line in file:
            if "time:" in line:
                time_str = line.split("time:")[1].split(",")[0].strip() 
                
                try:
                    timestamp = datetime.strptime(time_str, "%Y/%m/%d %H:%M:%S")
                    timestamps.append(timestamp)
                except ValueError:
                    continue  

        if len(timestamps) >= 2:
            first_time = timestamps[0]
            last_time = timestamps[-1]
            
            time_diff = last_time - first_time
            return time_diff
        else:
            return None 

def get_highest_train_accuracy_line(file_path):
    logs_path = os.path.join(file_path, "best_model.txt")
    max_accuracy = -float('inf')
    best_line = None
    with open(logs_path, 'r') as file:
        for line in file:
            match = re.search(r"test/accu:\s([\d.]+)", line)
            if match:
                accuracy = float(match.group(1))
                if accuracy > max_accuracy:
                    max_accuracy = accuracy
                    best_line = line.strip()

    return best_line

def extract_config(config_array):
  seperated_config_array = []
  for config in config_array:
      seperated_config_array.append(config.split(" ")[1])
  return seperated_config_array

def append_train_results_to_csv(results_array):
    csv_file_path = "./experimental_logs.csv"
    try:
        with open(csv_file_path, mode='a', newline='') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(results_array)
        print(f"Successfully appended array to {csv_file_path}: {results_array}")
    except Exception as e:
        print(f"An error occurred while appending to the CSV: {e}")


def perform_training():
    commands_path = "./commands.txt"
    errors_path = "./errors.txt"
    logdir = None
    config_array = []
    with open(commands_path, 'r') as file:
        current_command = []
        for line in file:
          line = line.strip()
          if not line or line.startswith('#'):
              continue
          
          if line.startswith('python3') or line.startswith("run"):
              if current_command:
                  try:
                    config_array = extract_config(current_command[1:])
                    logdir = "_".join([config_array[0], config_array[1]])
                    command = " ".join(current_command)
                    subprocess.run(command, shell=True, check=True)
                    highest_train_accu = get_highest_train_accuracy_line(logdir)
                    total_time = get_total_time(logdir)
                    _, total_memory, average_memory = extract_memory_cumsum(logdir)
                    final_accu = get_final_train_accu(logdir)

                    config_array.extend([total_time.__str__(), average_memory, total_memory, highest_train_accu, final_accu])
                    append_train_results_to_csv(config_array)
                    current_command = []
                  except subprocess.CalledProcessError as e:
                    with open(errors_path, 'a') as error_file:
                      error_file.write(f"Error executing command: {command}, {e}\n")
              if line.startswith("python3"):
                current_command = [line]
          else:
              current_command.append(line)

perform_training()
