# Prompts

This section contains the system prompts and templates used to instantiate the Hypostructure framework as a deterministic theorem-verification runtime within Large Language Models.

## Overview

The prompts transform an LLM from a "helpful assistant" into a **Hypostructure Diagnostic Engine** that mechanically executes the Structural Sieve Algorithm. Rather than treating the framework specification as context to summarize, the model treats it as source code to execute.

## Available Prompts

| Prompt | Purpose | Description |
|--------|---------|-------------|
| [System Prompt](system_prompt.md) | LLM Configuration | Core directive that initializes the Diagnostic Engine, defining operational semantics, execution loops, and behavioral inhibitors |
| [Universal Solution Template](template.md) | Proof Object Format | Complete template for generating machine-checkable proof objects, including interface permits, sieve execution, and certificate chains |

## Prompt Descriptions

### System Prompt

The **System Prompt** configures an LLM to operate as the Hypostructure Diagnostic Engine. Key components:

- **Core Directive:** Establishes the model as a deterministic theorem-verification runtime
- **Operational Semantics:** Defines the Thin Object Protocol, Logic Gate Protocol, and Metatheorem Override rules
- **Execution Loop:** Specifies the four phases: Instantiation, Sieve Run, Lock, and Result Extraction
- **Behavioral Inhibitors:** Suppresses hedging, historical references, and undecidable defaults

### Universal Solution Template

The **Universal Solution Template** provides the standard format for generating complete Hypostructure proof objects. It includes:

- **Metadata:** Problem specification, system type, and framework version
- **Automation Witness:** Certification of eligibility for Universal Singularity Modules
- **Interface Permits:** Complete checklist for all required certificates
- **Sieve Execution:** Node-by-node traversal with certificate emissions
- **Lock Mechanism:** Final verdict determination and certificate chain

## Usage

1. Copy the **System Prompt** into your LLM's system prompt field
2. Provide the framework specification (`hypopermits_jb.md`) as context
3. Use the **Universal Solution Template** as the output format for proof generation
4. Submit a problem definition to receive a machine-checkable proof object
