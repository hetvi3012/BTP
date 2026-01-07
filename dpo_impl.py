import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from torch.utils.data import DataLoader

# ==========================================
# 1. The DPO Loss Function (Equation 7)
# ==========================================
# This is the heart of the paper. We implement Eq 7 from scratch.
def dpo_loss(policy_chosen_logps, policy_rejected_logps, ref_chosen_logps, ref_rejected_logps, beta=0.1):
    """
    Computes the DPO loss for a batch of policy and reference model log probabilities.
    Args:
        policy_chosen_logps: Log probs of the policy model for the preferred answer.
        policy_rejected_logps: Log probs of the policy model for the dispreferred answer.
        ref_chosen_logps: Log probs of the reference model for the preferred answer.
        ref_rejected_logps: Log probs of the reference model for the dispreferred answer.
        beta: The 'beta' hyperparameter from the paper (controls deviation from reference).
    """
    # Calculate the log ratio for the Policy Model (pi_theta)
    pi_logratios = policy_chosen_logps - policy_rejected_logps
    
    # Calculate the log ratio for the Reference Model (pi_ref)
    ref_logratios = ref_chosen_logps - ref_rejected_logps

    # The core DPO logic: Difference of the log ratios
    logits = pi_logratios - ref_logratios

    # The loss is the negative log sigmoid of the scaled logits
    # Formula: -log(sigmoid(beta * (log(pi_w/ref_w) - log(pi_l/ref_l))))
    losses = -F.logsigmoid(beta * logits)
    
    # Return average loss, chosen rewards, and rejected rewards (for tracking)
    chosen_rewards = beta * (policy_chosen_logps - ref_chosen_logps).detach()
    rejected_rewards = beta * (policy_rejected_logps - ref_rejected_logps).detach()
    
    return losses.mean(), chosen_rewards.mean(), rejected_rewards.mean()

# ==========================================
# 2. Helper to Get Log Probabilities
# ==========================================
def get_log_probs(model, input_ids, attention_mask):
    """
    Runs the model and extracts log probabilities for the specific tokens in input_ids.
    """
    logits = model(input_ids, attention_mask=attention_mask).logits
    # Shift logits and labels so we predict the next token
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = input_ids[..., 1:].contiguous()
    
    # Compute Cross Entropy per token (without reduction)
    loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    
    # Sum up log probs for each sequence
    loss = loss.view(shift_labels.size())
    return -loss.sum(dim=1)

# ==========================================
# 3. Training Setup
# ==========================================
def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Starting DPO Training on {device}...")

    # Load Models (Policy and Reference)
    # Ideally pi_ref is frozen (not trained)
    model_name = "gpt2" # Using small model for testing
    policy_model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    ref_model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    ref_model.eval() # Freeze reference model
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # Dummy Data: A "Positive" (Chosen) and "Negative" (Rejected) review pair
    # In a real run, you load the Anthropic HH dataset here
    data = [
        {"chosen": "This movie was fantastic and I loved the acting.", 
         "rejected": "This movie was terrible and boring."},
        {"chosen": "The product works great and arrived on time.", 
         "rejected": "Broken on arrival, waste of money."}
    ] * 10 # Duplicate to simulate a batch

    optimizer = torch.optim.AdamW(policy_model.parameters(), lr=1e-5)

    print("Training Loop Starting...")
    for epoch in range(5):
        for batch in data:
            # Tokenize Inputs
            chosen_inputs = tokenizer(batch["chosen"], return_tensors="pt", padding=True, truncation=True).to(device)
            rejected_inputs = tokenizer(batch["rejected"], return_tensors="pt", padding=True, truncation=True).to(device)

            # 1. Get Log Probs from Policy Model (Gradient enabled)
            policy_chosen_logps = get_log_probs(policy_model, chosen_inputs["input_ids"], chosen_inputs["attention_mask"])
            policy_rejected_logps = get_log_probs(policy_model, rejected_inputs["input_ids"], rejected_inputs["attention_mask"])

            # 2. Get Log Probs from Reference Model (No Gradient)
            with torch.no_grad():
                ref_chosen_logps = get_log_probs(ref_model, chosen_inputs["input_ids"], chosen_inputs["attention_mask"])
                ref_rejected_logps = get_log_probs(ref_model, rejected_inputs["input_ids"], rejected_inputs["attention_mask"])

            # 3. Calculate DPO Loss
            loss, reward_chosen, reward_rejected = dpo_loss(
                policy_chosen_logps, policy_rejected_logps, 
                ref_chosen_logps, ref_rejected_logps
            )

            # 4. Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1} | Loss: {loss.item():.4f} | Reward Margin: {(reward_chosen - reward_rejected).item():.4f}")

    print("Training Complete. The model now prefers positive sentiments!")

if __name__ == "__main__":
    train()