### governance_and_risks.md

### why talk about governance in a churn system

Introducing a predictive model into a business process quickly raises questions that go beyond algorithmic performance alone. As soon as a system influences, even indirectly, decisions that have an impact on customers, it becomes necessary to question the rules that govern its use. In this project, governance is not viewed as an external or regulatory constraint, but as an intrinsic component of the system itself.

Choosing to consider the model as a decision support tool immediately implies a clear allocation of responsibilities. The system produces signals, proposes priorities, or highlights at-risk situations, but it does not decide. This distinction, which may appear formal, in fact conditions the entire set of control and supervision mechanisms put in place.

### feedback loop risks

One of the most frequently underestimated risks in churn systems concerns feedback loops. When a model is used to target certain customers, the actions taken on the basis of its predictions modify the future behavior of those customers, and therefore the data collected afterward. Over time, the model may end up being trained on data that reflect its own past decisions more than the original phenomenon it was intended to capture.

In the present case, an overly aggressive retention system could, for example, concentrate efforts on certain customer profiles, leaving other segments relatively unchanged. The risk is then to artificially reinforce certain correlations while masking others. This drift is not immediately visible in standard metrics, but it can gradually erode the relevance of the model.

To mitigate this risk, the system is designed to make explicit the decisions taken on the basis of predictions. Traceability of the thresholds used, the weights applied, and the actions undertaken constitutes a first lever of control. It allows, a posteriori, analysis of whether the observed changes in the data are compatible with the phenomenon under study or whether they result in part from the use of the system itself.

### model aging and data drift

Another central issue concerns model aging. The data used to train a churn model reflect a past situation at a given point in time. Yet commercial offerings, customer behaviors, and market conditions evolve. A model that appeared relevant at one point can gradually become less reliable, without this degradation being immediately detectable.

In this project, the model is not considered a stable artifact. Its use is conditional on implicit assumptions about the stability of the data and the relationships they contain. When these assumptions no longer hold, the system must be questioned. This may lead to retraining, but also, in some cases, to a temporary suspension of the model’s use.

System governance therefore includes the idea that non-use can be a valid decision. Continuing to use an obsolete model simply because it exists can prove more risky than temporarily reverting to non-automated business rules. This possibility is integrated from the design stage, in order to avoid excessive dependence on the tool.

### biases induced by system usage

Even when the initial data are relatively balanced and the model is technically sound, the use of the system can introduce new biases. Repeated targeting of certain profiles can modify their service experience, influence their behavior, or create unintended stigmatization effects. These biases do not necessarily originate from the model itself, but from the way its outputs are exploited.

It might be tempting to consider these effects as externalities that are difficult to control. However, a responsible governance approach consists in acknowledging their existence and taking them into account in the overall evaluation of the system. This involves, for example, monitoring the impact of retention actions on different customer segments, beyond churn indicators alone.

In this context, governance is not limited to static rules. It requires continuous observation of the system’s effects, and the ability to adjust, or even to challenge, certain practices when drifts are identified.

### usage rules and human responsibility

A fundamental principle of the system presented here is human responsibility in the final decision. The model does not provide a verdict, but contextualized probabilistic information. Usage rules specify that this information can only be used in conjunction with other elements, and that it cannot, on its own, justify an automatic action toward a customer.

This human responsibility is also linked to the ability to justify a decision. When a retention action is undertaken, it must be possible, at a minimum, to explain why the customer was targeted, which elements were taken into account, and to what extent the model influenced the decision. This requirement for justification helps to limit abusive or mechanical uses of the system.

Finally, it should be emphasized that governance does not aim to eliminate all risk-taking. Rather, it seeks to make these risks visible, discussable, and, as far as possible, manageable. In a real-world context, the complete absence of risk is illusory. By contrast, the absence of an explicit framework constitutes, in itself, a major risk.

### governance integrated into system design

The approach adopted in this project is based on the idea that governance should not be added a posteriori. Rules, limits, and supervision mechanisms are conceived at the same time as technical choices. This integration makes it possible to design a system whose uses are defined as clearly as its expected performance.

Rather than seeking to maximize the model’s autonomy, the system embraces its dependence on an organizational and human framework. Far from being a weakness, this dependence is a central element of its robustness. It makes it possible to embed the tool in a responsible decision-making practice, where technology remains in the service of clearly defined objectives that are regularly reassessed.
