#!/bin/bash

# 1. Get the current date and time (start time) - monotonic clock
start_time=$(date +%s%N | cut -b1-13)

# 2. Execute the Python script
python train.py

# 3. Get the current date and time (end time) - monotonic clock
end_time=$(date +%s%N | cut -b1-13)

# 4. Calculate the elapsed time in milliseconds.
elapsed_ms=$(echo "($end_time - $start_time)" | bc)

# 5. Calculate the elapsed time in seconds.
elapsed_seconds=$(echo "$elapsed_ms / 1000" | bc)

# 6. Print the elapsed time in seconds (with 4 decimal places)
printf "Elapsed time (seconds): %.4f\n" "$elapsed_seconds"

# 7. Convert elapsed seconds to HH:MM:SS format (using GNU date).
#    Check for empty and command existence
if [ -n "$elapsed_seconds" ] && command -v date > /dev/null 2>&1; then
    # Truncate to whole seconds before converting
    elapsed_seconds_int=$(printf "%.0f" "$elapsed_seconds")
    human_readable_time=$(date -u -d "@$elapsed_seconds_int" +"%H:%M:%S")
    echo "Elapsed time (HH:MM:SS): $human_readable_time"
fi

# --- Optional error handling (uncomment to use) ---
# python test.py
# if [ $? -ne 0 ]; then
#   echo "Error: Python script exited with a non-zero status." >&2
#   exit 1
# fi

# --- Optional logging (uncomment to use) ---
# log_file="execution_log.txt"
# echo "Start time: $(date -d @$(echo "$start_time / 1000" | bc) '+%Y-%m-%d %H:%M:%S.%3N')" >> "$log_file"
# echo "End time:   $(date -d @$(echo "$end_time / 1000" | bc) '+%Y-%m-%d %H:%M:%S.%3N')" >> "$log_file"
# printf "Elapsed time (seconds): %.4f\n" "$elapsed_seconds" >> "$log_file"
# echo "Elapsed time (HH:MM:SS): $human_readable_time" >> "$log_file" # Log human-readable time
# echo "------------------------------------" >> "$log_file"

# notify
python notify.py "DNDM in school training is done"