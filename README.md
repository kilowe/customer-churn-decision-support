# README.md

## customer churn decision support system (telco case)

This project is part of a broader reflection on the use of machine learning models in real-world business contexts, where prediction alone is only one element among others in a wider decision-making process. It is based on a classic customer churn case in the telecommunications sector, but the objective is not to produce a high-performing model in the strict sense, nor to optimize a classification score in isolation.

The work presented here instead seeks to design a decision support system that integrates a predictive component while taking into account business constraints, usage risks, and the need for explicit human oversight. In this framework, the model is never considered an autonomous agent, but rather an assistive tool intended to inform certain decisions without replacing them.

The project relies on the Telco Customer Churn dataset (IBM), which brings together contractual, demographic, and usage information for a set of customers. Although synthetic, these data make it possible to explore issues close to real industrial situations, particularly with regard to class imbalance, costly decisions, and the temporal dynamics of the phenomenon under study.

Particular attention is paid to the distinction between prediction and decision. The output of a probabilistic model is never interpreted as an operational truth, but as a signal subject to explicit rules, notably through the choice of thresholds, the analysis of false negatives and false positives, and the introduction of a human validation mechanism. This approach leads to considering the system as a whole rather than the model in isolation.

From a technical standpoint, the project includes a data preparation pipeline, the training of simple and interpretable classification models, and a module dedicated to the analysis of thresholds and class weightings. These technical choices do not aim at algorithmic sophistication, but at the readability, traceability, and governability of the system being built.

Beyond the technical aspects, the project incorporates an explicit reflection on the systemic risks associated with the use of such a system. In particular, it addresses issues such as feedback loops, model aging, biases induced by decisions made on the basis of predictions, as well as the conditions under which the system should not be used. These elements are documented separately in order to make visible the limitations and underlying assumptions.

This repository should therefore be read as the description of an AI-based decision support system that is in the process of being designed and analyzed, rather than as a standard academic machine learning exercise. It reflects an exploratory approach, grounded in practical concerns, where human responsibility and business understanding occupy a central place.

### Repository structure (indicative)

- `src/` : data preparation, training, and decision analysis scripts
- `data/` : raw data and preprocessed data
- `docs/` : documents describing the business context, decision logic, governance, and system limitations

### Project positioning

This work is part of an approach focused on the integration and supervision of AI systems, rather than on the development of isolated models. It aims to illustrate how machine learning tools can be responsibly integrated into existing business processes, with explicit usage rules and clearly defined human responsibility.

## Reproducibility supplement

The original scripts in `src/` provide the initial prototype for preprocessing, model training, threshold comparison, and testing.

This supplement clarifies the workflow by making the train/validation/test separation explicit and by documenting threshold selection and business-cost evaluation in a transparent sequence.

The supplement does not replace the original scripts; it clarifies and extends them to support reproducibility and auditability.

### Execution order

Run the full supplement with:

```bash
python src/reproducibility_supplement/run_reproducibility_supplement.py
```

### Generated outputs

The supplement writes outputs to `outputs/reproducibility_supplement/`:

- `split_summary.csv`
- `validation_probabilities.csv`
- `test_probabilities.csv`
- `threshold_grid_validation.csv`
- `selected_thresholds.csv`
- `Evaluation_Results.csv`
- `Business_Cost_Analysis.csv`
- `Final_Summary.csv`
- `DSS_Risk_Output.csv`
- `risk_tier_summary.csv`
