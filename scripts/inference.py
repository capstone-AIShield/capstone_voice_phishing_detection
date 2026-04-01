import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_dir = os.path.join(base_dir, "models", "SDQ-LLM-LoRA")
    
    if not os.path.exists(model_dir):
        print(f"Error: Trained model not found at {model_dir}. Please run train.py first.")
        return

    print("Loading Base Model and Tokenizer...")
    base_model_name = "skt/kogpt2-base-v2"
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    base_model = AutoModelForCausalLM.from_pretrained(base_model_name)

    print("Applying LoRA Weights...")
    model = PeftModel.from_pretrained(base_model, model_dir)
    
    model.eval()

    test_scenarios = [
        "명령어: 다음은 카드 배송원 사칭형 보이스피싱 의심 상황입니다. 적절한 대처 방안을 제시하세요.\n답변:",
        "명령어: 상황: 모르는 번호로 결제 승인 문자가 오고 전화를 유도합니다. 해당 보이스피싱 수법에 대처하는 안내 스크립트를 출력해줘.\n답변:",
        "명령어: 신분증 노출, 악성앱 설치 등으로 개인정보가 유출되었을 경우 대처 방법을 알려주세요.\n답변:"
    ]

    print("\n" + "="*50)
    print("SDQ-LLM INFERENCE TEST")
    print("="*50)

    for prompt in test_scenarios:
        inputs = tokenizer(prompt, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                max_new_tokens=100,
                temperature=0.3, # Low temperature for factual response
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.2
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"\n[INPUT]\n{prompt}")
        print(f"[OUTPUT]\n{response.replace(prompt, '').strip()}")

if __name__ == "__main__":
    main()
