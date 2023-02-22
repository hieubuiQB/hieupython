from parrot import Parrot
import torch
import warnings

warnings.filterwarnings("ignore")

def set_random_seed(seed: int):
    """
    Set random seed for reproducibility.
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_random_seed(1234)

# Init model
parrot = Parrot(model_tag="prithivida/parrot_paraphraser_on_T5")

phrases = [
    "Do you like dogs",
    "Do you like to eat insect sausages?"
]

for phrase in phrases:
    print("-" * 100)
    print("Input phrase: ", phrase)
    print("-" * 100)
    para_phrases = parrot.augment(input_phrase=phrase, use_gpu=False)
    if para_phrases:
        for para_phrase in para_phrases:
            print(para_phrase)
else:
    print("No paraphrases generated for the input phrase.")