try:
    import transformers
    import torch
    import pandas
    from PIL import Image

    print("✅ All imports successful. You're good to go!")
except Exception as e:
    print("❌ Error:", e)
