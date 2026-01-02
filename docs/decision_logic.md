### decision_logic.md

### from prediction to decision

In many machine learning projects applied to churn, the model output is often presented as an answer in itself. A customer is classified as “at risk” or “not at risk,” sometimes accompanied by a probability, and then directly associated with an implicit action. This simplification, while operational in certain contexts, becomes problematic as soon as one attempts to integrate the model into a real business process.

In this project, prediction is never considered a decision. It constitutes an intermediate signal whose value depends entirely on how it is interpreted, filtered, and combined with explicit rules. In other words, the core of the system does not lie solely in the classification model, but in the logic that transforms a probability into a proposed course of action.

The model used produces an estimate of churn risk for each customer. This estimate is continuous and generally falls between 0 and 1. At this stage, no decision has yet been made. The score simply indicates that a customer exhibits a profile more or less similar to those who churned in the past, according to the available data. It should nevertheless be kept in mind that this statistical proximity does not directly imply the customer’s actual intention nor the relevance of an intervention.

### the central role of the threshold

The transformation of the score into a decision signal relies on the choice of a threshold. This threshold is not treated here as a secondary technical parameter, but as a genuine business lever. Setting a threshold amounts to deciding from which level of risk a customer deserves special attention, knowing that this attention entails both financial and organizational costs.

A low threshold makes it possible to identify a large number of potentially at-risk customers, but mechanically increases the number of false positives. Conversely, a high threshold reduces unnecessary interventions, at the cost of an increased risk of letting truly churning customers leave. There is no universally optimal threshold, as this choice depends heavily on the retention strategy, available resources, and the organization’s tolerance for risk.

In this context, the analysis of false negatives and false positives becomes central. False negatives correspond to customers who leave without having been identified as at risk. They are often perceived as more costly, as they represent a net loss with no retention attempt. False positives, on the other hand, mobilize resources for customers who would not have left the service, potentially generating indirect costs or a degradation of the customer experience.

Rather than seeking to maximize a global metric such as accuracy, the system is therefore evaluated in light of these differentiated errors. The threshold is adjusted according to business scenarios, taking into account the relative impact of the different types of errors. This approach necessarily involves trade-offs, and it would be illusory to claim that they can be completely eliminated through algorithmic optimization.

### class weighting as a strategic lever

The class imbalance present in churn data poses a classic challenge in supervised classification. The majority of customers do not churn, which can lead a naïve model to favor the majority class. In this project, class weighting is not approached merely as a technique to improve model performance, but as a means of steering its behavior in line with business priorities.

Assigning a higher weight to the churner class implicitly signals that errors on this class are more costly. This decision is not neutral. It modifies the model’s decision boundary and influences the distribution of the scores produced. Here again, this is not an automatic optimization, but a strategic choice that must be understood and accepted.

It is important to note that class weighting and threshold selection interact. A model trained with adjusted weights will produce different scores, which may require reconsideration of the decision threshold. This interdependence reinforces the idea that decision logic cannot be reduced to a sequence of independent steps. It must be conceived as a coherent whole, in which each parameter influences the others.

### human validation and non-automation of the decision

A key aspect of the logic adopted here concerns the role given to human intervention. The system does not automatically trigger retention actions. It produces recommendations or alerts, which are then subject to human validation. This validation can take different forms depending on the organization, but it constitutes a deliberate barrier against excessive automation.

This approach is based on the idea that certain pieces of information relevant to the decision are not captured by the available data. A human agent may, for example, have access to recent contextual elements, qualitative feedback, or specific constraints that escape the model. The system is therefore designed to assist the decision, not to replace it.

One might object that this human intervention reduces operational efficiency. From an empirical standpoint, it primarily helps to limit serious errors and to maintain clearly assigned responsibility. In a system where decisions have direct consequences for customers, this responsibility cannot be fully delegated to a statistical model.

### explicit and documented decision logic

All of these choices aim to make the decision logic as explicit as possible. Thresholds, weights, and usage rules are not hidden behind technical abstractions, but documented as central elements of the system. This transparency not only facilitates a posteriori analysis of the decisions made, but also the future evolution of the system.

Ultimately, the logic presented here does not seek to eliminate the uncertainty inherent in churn. Rather, it seeks to frame it by providing a clear decision-making framework in which the model’s limitations are acknowledged and integrated. It is in this recognition, more than in raw performance, that the robustness of the proposed system resides.
