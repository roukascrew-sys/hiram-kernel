from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set


@dataclass(frozen=True)
class Rule:
    rule_name: str
    premises: List[str]
    conclusion: str


@dataclass
class DerivationStep:
    step_id: int
    rule_name: str
    required_premises: List[str]
    derived_conclusion: str


@dataclass
class ProofCertificate:
    target_claim: str
    steps: List[DerivationStep] = field(default_factory=list)


def verify_certificate(
    certificate: ProofCertificate,
    known_facts: Set[str],
    authorized_rules: Dict[str, Rule],
) -> bool:
    """Deterministically audits a derivation certificate in strict O(Steps) time.

    Enforces rule fidelity, premise grounding, and acyclic ordering.
    """
    derived_facts: Set[str] = set(known_facts)

    if not certificate.steps:
        # An empty proof cannot derive a claim unless it is already an established axiom
        return certificate.target_claim in derived_facts

    for step in certificate.steps:
        # 1. Rule Authorization Check
        if step.rule_name not in authorized_rules:
            return False

        canonical_rule = authorized_rules[step.rule_name]

        # 2. Rule Premise Fidelity Check (Prevent Premise-Stripping Attacks)
        if set(step.required_premises) != set(canonical_rule.premises):
            return False

        # 3. Rule Conclusion Fidelity Check (Prevent Conclusion-Swapping Attacks)
        if step.derived_conclusion != canonical_rule.conclusion:
            return False

        # 4. Grounding Check: All required premises must be satisfied by axioms or earlier steps
        for premise in step.required_premises:
            if premise not in derived_facts:
                return False

        # 5. Monotonic fact derivation
        derived_facts.add(step.derived_conclusion)

    # 6. Target Goal Check
    return certificate.target_claim in derived_facts


def prove_claim(
    target_claim: str,
    known_facts: Set[str],
    authorized_rules: Dict[str, Rule],
) -> Optional[ProofCertificate]:
    """Forward-chaining prover: searches for a path from known_facts to target_claim.

    Emits a ProofCertificate if proven; returns None if unprovable.
    """
    if target_claim in known_facts:
        return ProofCertificate(target_claim=target_claim, steps=[])

    derived_facts: Set[str] = set(known_facts)
    steps: List[DerivationStep] = []
    step_counter = 1

    while True:
        progress_made = False

        for rule in authorized_rules.values():
            if rule.conclusion in derived_facts:
                continue

            # Ensure ALL premises are satisfied before firing the rule
            if all(premise in derived_facts for premise in rule.premises):
                steps.append(
                    DerivationStep(
                        step_id=step_counter,
                        rule_name=rule.rule_name,
                        required_premises=list(rule.premises),
                        derived_conclusion=rule.conclusion,
                    )
                )
                step_counter += 1
                derived_facts.add(rule.conclusion)
                progress_made = True

                if rule.conclusion == target_claim:
                    return ProofCertificate(
                        target_claim=target_claim, steps=steps
                    )

        if not progress_made:
            return None