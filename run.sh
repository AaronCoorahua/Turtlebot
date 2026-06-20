function run
  source venv/bin/active.fish
  python main.py 2>&1 | tee logs/log_(date +%Y%m%d_%H%S).txt
end
