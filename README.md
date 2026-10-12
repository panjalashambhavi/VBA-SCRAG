# VBA-SCRAG — Verify-before-Accept Self-Correcting RAG

<p align="center">
  <strong>A Reliability-Aware Framework for Evaluating and Accepting RAG Corrections</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/AI-Retrieval--Augmented%20Generation-blue" alt="RAG">
  <img src="https://img.shields.io/badge/Framework-PyTorch-red" alt="PyTorch">
  <img src="https://img.shields.io/badge/Embeddings-Sentence--Transformers-orange" alt="Sentence Transformers">
  <img src="https://img.shields.io/badge/Status-Research%20Prototype-yellow" alt="Research Prototype">
</p>

> **VBA-SCRAG** investigates a simple but important question: *Should a RAG system automatically accept a corrected answer, or should it verify the candidate before replacing the original answer?*
>
> The framework introduces an acceptance gate that compares the grounding scores of an initial answer and a correction candidate. The project evaluates whether this mechanism improves answer reliability using controlled experiments.

---

## 1. Why VBA-SCRAG?

Retrieval-Augmented Generation (RAG) combines document retrieval with language generation to produce answers grounded in external evidence. However, retrieving documents does not guarantee that the generated answer is factually correct.

A conventional RAG pipeline follows this process:

```text
User Question
      |
      v
Document Retrieval
      |
      v
Answer Generation
      |
      v
Generated Answer
```

A self-correcting system adds another step: it attempts to improve an answer when the initial response appears insufficient. However, **a correction is not necessarily better than the original answer**.

### The Research Problem

- **Incorrect initial answers:** Retrieved evidence may not be sufficient to produce a correct response.
- **Unreliable corrections:** A regenerated answer can remain incorrect or become worse.
- **Weak verification signals:** Semantic similarity between an answer and retrieved evidence does not necessarily establish factual correctness.
- **Uncontrolled acceptance:** Automatically replacing an answer with a correction may introduce additional errors.

### The VBA-SCRAG Approach

VBA-SCRAG investigates a verify-before-accept strategy:

1. Generate an initial answer from retrieved evidence.
2. Calculate an initial grounding score.
3. Attempt correction when the score is below a configured threshold.
4. Generate a correction candidate using reformulated instructions and newly retrieved evidence.
5. Compare the candidate's grounding score with the initial score.
6. Accept the candidate only when it satisfies the configured improvement margin; otherwise, retain the initial answer.

The goal is to investigate whether selective acceptance can improve reliability compared with accepting every correction candidate.

---

## 2. System Architecture

```text
                  User Question
                        |
                        v
                +----------------+
                |    Retriever   |
                |  Top-k = 3     |
                +----------------+
                        |
                        v
                +----------------+
                | Answer         |
                | Generator      |
                +----------------+
                        |
                        v
                +----------------+
                | Grounding      |
                | Verifier       |
                +----------------+
                        |
                        v
                +----------------+
                | Initial Score  |
                | Below 0.75?    |
                +----------------+
                   /          \
                 No            Yes
                 |              |
                 v              v
          Retain Initial   Corrector and
              Answer       New Retrieval
                                |
                                v
                         Candidate Answer
                                |
                                v
                         Candidate Grounding
                                |
                                v
                         +------------------+
                         | Acceptance Gate  |
                         | Improvement >=   |
                         | 0.001            |
                         +------------------+
                            /          \
                          Yes           No
                           |             |
                           v             v
                      Accept New    Retain Initial
                        Answer         Answer
```

### Main Components

| Component | Implementation |
|---|---|
| Retriever | Sentence Transformer embeddings with top-3 retrieval |
| Embedding model | `all-MiniLM-L6-v2` |
| Answer generator | `google/flan-t5-small` |
| Grounding verifier | Cosine similarity between answer and retrieved evidence embeddings |
| Corrector | Adds a factual-precision instruction before retrieving and generating a candidate |
| Acceptance controller | Accepts a candidate when its grounding score improves by at least 0.001 |
| Generation configuration | `do_sample=False` |

### Grounding Score

The current verifier embeds the generated answer and each retrieved evidence document. It uses the maximum cosine similarity as the grounding score:

\[
G(a,E)=\max_{e\in E}\operatorname{cosine}(a,e)
\]

where \(a\) is the answer and \(E\) is the set of retrieved evidence documents.

The acceptance condition is:

\[
G(a_c,E_c)-G(a_0,E_0)\geq 0.001
\]

Here, \(a_0\) is the initial answer and \(a_c\) is the correction candidate.

**Important limitation:** This score measures semantic alignment, not calibrated confidence or verified factual correctness.

---

## 3. Experimental Evaluation

### Experimental Setup

| Parameter | Configuration |
|---|---|
| Dataset | HotpotQA distractor validation split |
| Pilot sample | 50 questions |
| Retriever and embedding model | `all-MiniLM-L6-v2` |
| Generator | `google/flan-t5-small` |
| Number of retrieved documents | 3 |
| Correction threshold | 0.75 |
| Minimum grounding improvement | 0.001 |
| Generation sampling | Disabled (`do_sample=False`) |
| Answer-quality metrics | Answer F1 and Exact Match |
| Additional metric | Semantic answer similarity |

Three systems are compared:

- **Vanilla RAG:** Generates an answer without correction.
- **Correction without gate:** Accepts a correction candidate whenever correction is triggered.
- **VBA-SCRAG:** Accepts a correction candidate only when the grounding improvement satisfies the gate.

### 3.1 Compared Answer-Quality Results

| System | Answer F1 | Exact Match | Semantic Similarity |
|---|---:|---:|---:|
| Vanilla RAG | 28.96% | 22.00% | 0.5409 |
| Correction without gate | **34.84%** | **24.00%** | **0.5814** |
| VBA-SCRAG | 31.12% | 22.00% | 0.5573 |

*Results are from the 50-question pilot evaluation. Semantic similarity measures meaning-relatedness to the reference answer; it does not guarantee factual correctness.*

### 3.2 Paired Acceptance-Gate Experiment

To investigate the gate more directly, the paired experiment generated one initial answer and one correction candidate per question. Both gated and ungated systems used the same initial answers and correction candidates; only the acceptance decision differed.

| Measurement | Correction without gate | VBA-SCRAG |
|---|---:|---:|
| Correction attempts | 45 | 45 |
| Candidates accepted | 45 | 3 |
| Candidates rejected | 0 | 42 |
| Mean final grounding score | 0.334418 | **0.354460** |
| Answer F1 | **34.84%** | 31.12% |
| Exact Match | **24.00%** | 22.00% |

### Key Finding

The acceptance gate increased mean grounding similarity compared with correction without a gate. However, Answer F1 and Exact Match were lower for VBA-SCRAG in this pilot experiment.

Inspection of the three accepted corrections showed that none improved Answer F1, despite their higher grounding scores.

**Interpretation:** The current gate changes correction acceptance decisions and improves the measured grounding score, but the experiments do not establish that it improves factual answer correctness.

These findings motivate a stronger evidence-based verification mechanism.

---

## 4. Repository Structure

```text
VBA-SCRAG/
│
├── src/
│   ├── baselines.py
│   ├── correction.py
│   ├── evaluation.py
│   ├── generator.py
│   ├── retriever.py
│   ├── verifier.py
│   ├── vba_scrag.py
│   └── vba_scrag_fixed.py
│
├── results/
│   ├── paired_gate_experiment.csv
│   ├── fixed_system_comparison.csv
│   ├── fixed_threshold_ablation.csv
│   ├── fixed_vanilla_rag_correctness.csv
│   ├── fixed_self_correcting_rag_correctness.csv
│   ├── fixed_vba_scrag_correctness.csv
│   └── ... additional experiment outputs
│
├── analyze_paired_results.py
├── check_results.py
├── evaluate_correctness.py
├── evaluate_semantic_similarity.py
├── inspect_accepted_corrections.py
├── run_experiment.py
├── run_experiment_fair.py
├── run_paired_gate_experiment.py
├── threshold_ablation_experiment.py
├── threshold_ablation_fair.py
├── paper_overleaf_source.txt
├── README.md
└── .gitignore
```

---

## 5. Installation and Quickstart

### Prerequisites

- Python installed on your system
- Git
- Internet access for downloading the required models and dataset

### Clone the Repository

```bash
git clone https://github.com/panjalashambhavi/VBA-SCRAG.git
cd VBA-SCRAG
```

### Create a Virtual Environment

**Windows CMD:**

```cmd
python -m venv venv
venv\Scripts\activate
```

**Linux/macOS:**

```bash
python -m venv .venv
source .venv/bin/activate
```

### Install Dependencies

Install the libraries used by the project:

```bash
pip install torch transformers sentence-transformers numpy pandas scikit-learn datasets
```

The project may require additional dependencies depending on the selected script and the current environment.

### Run Experiments

Run the main comparison:

```bash
python run_experiment_fair.py
```

Run the paired acceptance-gate experiment:

```bash
python run_paired_gate_experiment.py
```

Run the threshold ablation:

```bash
python threshold_ablation_fair.py
```

Evaluate answer correctness:

```bash
python evaluate_correctness.py
```

Evaluate semantic answer similarity:

```bash
python evaluate_semantic_similarity.py
```

Analyze paired results:

```bash
python analyze_paired_results.py
```

Check the individual script's configuration and output paths before running it. Model downloads and experiment execution may take time, particularly on CPU-only systems.

---

## 6. Current Limitations

The current implementation has several limitations:

1. **Embedding similarity is an imperfect verifier.** A higher grounding score does not guarantee that the answer is factually correct.
2. **Limited evaluation size.** The current pilot uses 50 HotpotQA questions, so the results may not generalize to larger datasets.
3. **Answer-quality results do not establish superiority.** In the paired experiment, the ungated system achieved higher Answer F1 and Exact Match.
4. **Deterministic generation is not a complete reproducibility guarantee.** Disabling sampling reduces one source of variability, but other sources of variability may remain.
5. **Limited correction strategy.** The current corrector uses a fixed instruction rather than an adaptive, multi-step reasoning strategy.

These limitations are part of the motivation for further experiments.

---

## 7. Future Work and Improvement Plan

### Phase 1 — Evidence-Based Verification

Investigate Natural Language Inference (NLI) to assess whether retrieved evidence supports, contradicts, or is neutral toward claims in a candidate answer.

### Phase 2 — Improved Acceptance Rule

Explore combining evidence-support signals with grounding-score changes instead of relying on embedding similarity alone.

### Phase 3 — Controlled Evaluation

Repeat paired experiments using the same initial answers and correction candidates across acceptance policies. Expand the evaluation sample and, if feasible, include another dataset.

### Phase 4 — Comprehensive Analysis

Evaluate Answer F1, Exact Match, grounding, evidence support, acceptance and rejection counts, runtime, and uncertainty estimates.

### Phase 5 — Research Paper Revision

Update the methodology, experimental results, failure analysis, limitations, and conclusion based on the measured outcomes.

**Current status:** The initial implementation and pilot evaluation are complete. NLI-based verification and expanded evaluation are planned future work, not completed features.

---

## 8. Author

**Panjala Shambhavi**  
B.Tech — Computer Science and Engineering (Artificial Intelligence and Machine Learning)

Research area: Retrieval-Augmented Generation, LLM reliability, evidence-based verification, and self-correcting AI systems.

---

## 9. Acknowledgment

This repository documents an ongoing research prototype. The reported results represent preliminary experiments, and further verification and evaluation are required before drawing conclusions about general reliability improvements.

