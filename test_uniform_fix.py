#!/usr/bin/env python3

from llm_qus_analyzer.chunker.models import QUSComponent
from llm_qus_analyzer.chunker.parser import Template
from llm_qus_analyzer.set.uniform import UniformAnalyzer

def create_test_component(text, role, means, ends, template_text):
    """Helper to create test components with mock templates."""
    template = Template(
        text=template_text,
        chunk={
            '[ROLE]': ['[NOUN]'] if role else [],
            '[MEANS]': ['[VERB]'] if means else [],
            '[ENDS]': ['[SCONJ]'] if ends else []
        },
        tail=None,
        order=['[ROLE]', '[MEANS]', '[ENDS]'] if ends else ['[ROLE]', '[MEANS]']
    )
    
    return QUSComponent(
        text=text,
        role=role,
        means=means,
        ends=ends,
        template=template
    )

def test_uniform_analyzer():
    print("Testing Uniform Analyzer with Template Text...")
    
    # Create test components with same template structure
    components = [
        create_test_component(
            "As a user, I want to login so that I can access my account",
            ["user"], "want to login", "I can access my account",
            "As a {ROLE}, I {MEANS} so that {ENDS}"
        ),
        create_test_component(
            "As a admin, I want to manage users so that I can control access",
            ["admin"], "want to manage users", "I can control access", 
            "As a {ROLE}, I {MEANS} so that {ENDS}"
        ),
        # This one should have a violation (different template)
        create_test_component(
            "User wants to delete account",
            ["User"], "wants to delete account", None,
            "{ROLE} {MEANS}"
        ),
        # This one has no ENDS (should show template without ENDS)
        create_test_component(
            "As a customer, I want to buy products",
            ["customer"], "want to buy products", None,
            "As a {ROLE}, I {MEANS}"
        )
    ]
    
    # Run uniform analysis
    results = UniformAnalyzer.run(None, 0, components)
    
    print(f"\nAnalyzed {len(components)} components:")
    for i, (violations, _) in enumerate(results):
        comp = components[i]
        print(f"\nComponent {i+1}: {comp.text[:50]}...")
        print(f"  Role: {comp.role}, Means: {comp.means}, Ends: {comp.ends}")
        print(f"  Template: {comp.template.text}")
        
        if violations:
            print(f"  VIOLATIONS ({len(violations)}):")
            for v in violations:
                print(f"    - {v.description}")
                print(f"    - Suggestion: {v.suggestion}")
        else:
            print("  ✅ No violations")

if __name__ == "__main__":
    test_uniform_analyzer()