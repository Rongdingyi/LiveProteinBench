python chat.py \
  --tasks pathway temperature \
  --models claude-3-7-sonnet-20250219 \
  --data-dir ./dataset/QA \
  --prompt-file ./prompt.json \
  --temperature 0.2 \
  --output-root ./responses
