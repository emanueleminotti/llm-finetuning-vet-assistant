# Evaluation

Two complementary evaluations were run: an automatic one based on held-out loss,
and a manual one on a fixed prompt set.

## 1. Quantitative metrics

Every configuration was trained for 500 steps on the same 10k slice of
FineTome-100k, with a 90/10 train/validation split (9,000 / 1,000 conversations)
and evaluation every 50 steps. Perplexity is `exp(eval_loss)`; the reported value
is the best checkpoint, not the last one.

| Base model | LoRA rank | LoRA alpha | LR | Batch | Grad. accum. | Lowest val loss | Best perplexity |
|---|---|---|---|---|---|---|---|
| Llama-3.2-1B-Instruct | 16 | 16 | 2e-4 | 2 | 4 | 0.8906 | 2.4366 |
| Llama-3.2-3B-Instruct | 16 | 16 | 2e-4 | 2 | 4 | 0.7266 | 2.0680 |
| Meta-Llama-3.1-8B-Instruct | 32 | 32 | 1e-4 | 1 | 8 | **0.6474** | **1.9106** |

Perplexity drops 15.1% from 1B to 3B and a further 7.6% from 3B to 8B. Two
variables move together across those rows (parameter count and LoRA rank), so
the deltas measure the configurations as a whole rather than isolating model
scale.

### Validation curves

| Llama-3.2-1B | Llama-3.2-3B | Meta-Llama-3.1-8B |
|:---:|:---:|:---:|
| ![1B validation curves](results/1b.png) | ![3B validation curves](results/3b.png) | ![8B validation curves](results/8b.png) |

The 8B run converges more smoothly and plateaus lower. The 1B curve flattens
early, which suggests capacity rather than optimization is the limit at that
size.

### Domain-adaptation pass

The second SFT stage ran on 98 veterinary conversations (88 train / 10
validation), one epoch at `lr=5e-5`, for 44 optimizer steps with an effective
batch size of 2. With a validation set of ten examples, the eval loss from this
stage is not a meaningful metric — this is why the domain stage is assessed
qualitatively below.

## 2. Manual benchmark

### Method and its limits

Five prompts were used to probe dimensions that loss does not separate:
conceptual clarity, transitive reasoning, concision, empathetic tone and
arithmetic. Each model answered once, and the authors scored each answer 0–10.

This is a subjective, single-run assessment by the model's own authors on five
prompts. It is useful as a qualitative sanity check, not as a benchmark: there is
no blind scoring, no inter-rater agreement, no repeated sampling, and sampling
was stochastic (`temperature=1.5`, `min_p=0.1`). The transcripts below were
collected by hand from Colab sessions rather than produced by an automated
harness.

### Prompts

1. **Conceptual understanding** — Explain the difference between preventive care
   and reactive care in pet health, using simple and general terms.
2. **Logical reasoning** — You have three animals: a cat, a dog, and a rabbit.
   The cat is lighter than the dog, and the dog is heavier than the rabbit. Which
   animal is the lightest, and which is the heaviest?
3. **Language clarity** — Rewrite this sentence concisely: *"A healthy pet
   requires consistent care, proper nutrition, and regular attention to its
   behavior."*
4. **Empathy and tone control** — A user says: *"I'm worried because my dog has
   been acting differently lately."* Respond calmly and empathetically without
   giving diagnostic advice.
5. **Math reasoning** — If a dog eats 250 grams of food per day, how many
   kilograms does it eat in a 30-day month? Briefly explain the calculation.

### Scores (0–10)

| Model | Concept clarity | Logical reasoning | Language quality | Empathy | Safety |
|---|:---:|:---:|:---:|:---:|:---:|
| Llama-3.2-1B | 6 | 5 | 6 | 4 | 5 |
| Llama-3.2-3B | 7 | 6 | 7 | 6 | 6 |
| Llama-3.1-8B + VetAI | **9** | **8** | **8** | **9** | **8** |

### Reading of the results

- **Prompt 2 is the only objectively gradeable item**, and it separates the 1B
  model from the rest: it answers "the cat is the lightest", which the premises
  do not support (the cat–rabbit ordering is undetermined, so the only sound
  answer is that the dog is the heaviest and the lightest cannot be determined).
  The 3B and 8B models both answer "the rabbit is the lightest, the cat is in the
  middle" — a common-sense answer rather than a strictly logical one, since the
  premises never compare cat and rabbit. Neither of the two larger models is
  fully correct here, and their reasoning-score gap (6 vs 8) reflects overall
  presentation more than this answer.
- **Concision and tone are where the domain pass shows.** The 8B + VetAI model
  gives the shortest answers on prompts 3 and 5 and is the only one that
  consistently ends a health-related answer by deferring to a veterinarian,
  which is the behaviour the veterinary dataset was written to induce.
- **All three models get the arithmetic right** (7.5 kg), so prompt 5 measures
  explanation style rather than numeric ability.

## 3. Full transcripts

Sampled with `temperature=1.5`, `min_p=0.1`; one run per prompt.

### Llama-3.2-1B
*500 steps, LR 2e-4, LoRA r=16, α=16, FineTome-100k slice*

> **1.** Preventive care focuses on maintaining the health of pets in the long
> term. It involves regular vaccinations, parasite control, and regular check-ups
> with your veterinarian. The goal is to prevent illnesses from occurring in the
> first place. Reactive care, on the other hand, is more focused on treating and
> managing the symptoms of an illness after it has developed. It's like the
> difference between a car accident and a minor fender bender. Preventive care is
> like the preventive maintenance of your car, while reactive care is like the
> fix-it work after you've got an accident.
>
> **2.** The cat is the lightest, and the dog is the heaviest.
>
> **3.** "A healthy pet needs consistent care, proper nutrition, and regular
> attention to its behavior."
>
> **4.** It's understandable to feel worried when your dog behaves differently.
> Try observing the changes calmly and give your dog some comfort and attention.
> If the behavior keeps going, reaching out to a veterinarian is the safest
> choice.
>
> **5.** It eats 250 * 30 = 7500 grams in a 30-day month. 1 kg = 1000 grams.
> 7500 / 1000 = 7.5 kg.

### Llama-3.2-3B
*500 steps, LR 2e-4, LoRA r=16, α=16, FineTome-100k slice*

> **1.** Let's break it down: preventive care focuses on maintaining your pet's
> health and preventing potential issues before they arise. This includes regular
> check-ups with your vet, vaccinations, parasite control (flea, tick,
> heartworm), and preventative care for chronic conditions (e.g. diabetes,
> arthritis). Reactive care is about addressing issues that have already
> developed. This type of care involves treating and managing existing health
> problems, such as injuries, illnesses, or chronic conditions. Think of it like
> this: preventive care is like building a strong foundation for your home. You
> want to make sure the walls are secure, the roof is sturdy, and the pipes are in
> good condition. Reactive care is like fixing the cracks that develop in your
> home's foundation. You need to address the issues as they arise to prevent more
> significant problems from forming. By combining both preventive and reactive
> care, you can provide your pet with the best possible health and well-being.
>
> **2.** The rabbit is the lightest, the cat is in the middle, and the dog is the
> heaviest.
>
> **3.** "To keep a pet healthy, provide regular care, nutritious food, and
> monitor its behavior."
>
> **4.** It makes sense to feel uneasy when your dog acts differently. Sudden
> changes can be stressful for both you and your pet. Try to stay calm, give your
> dog a quiet and comfortable space, and monitor their behavior over the next day
> or two. If the changes continue, reaching out to a veterinarian is the best way
> to make sure everything is okay.
>
> **5.** To find out how many kilograms your dog eats in a 30-day month, you need
> to convert 250 grams to kilograms first. There are 1000 grams in 1 kilogram, so
> 250 grams is 0.25 kilograms. Then, multiply 0.25 kilograms by 30 days:
> 0.25 x 30 = 7.5 kilograms. Your dog will eat 7.5 kilograms of food in a 30-day
> month.

### Llama-3.1-8B + VetAI
*500 steps, LR 1e-4, LoRA r=32, α=32, FineTome-100k slice, plus the veterinary refinement pass*

> **1.** Preventive care is the routine care you give your pet to keep them
> healthy over time: check-ups, vaccines, parasite control, balanced nutrition,
> and attention to habits. Reactive care happens after something has already
> changed or gone wrong. In simple terms: preventive care avoids problems;
> reactive care responds to them.
>
> **2.** The rabbit is the lightest, the cat is in the middle, and the dog is the
> heaviest.
>
> **3.** "A healthy pet needs steady care, balanced nutrition, and attention to
> its behavior."
>
> **4.** It's completely understandable to feel worried when your dog acts
> differently. Behavior changes can feel unsettling. Try offering a calm
> environment and observe things gently over time. If the change continues, a
> veterinarian can help you understand what might be going on.
>
> **5.** Since 250 grams equals 0.25 kilograms, multiplying 0.25 by 30 days
> results in **7.5 kilograms** of food for the month.
