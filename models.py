"""Pydantic schemas for extracting structured clinical data from a transcript.

These models are the target schema for structured output. LangChain/LangGraph
support this natively:

  - Agent:  `create_agent(model=..., response_format=ClinicalNote)`, then read
            the validated instance from `result["structured_response"]`.
  - Model:  `llm.with_structured_output(ClinicalNote)` for a bare chat model.

The Field descriptions double as extraction hints for the model, so keep them
clear and specific.
"""

from enum import Enum

from pydantic import BaseModel, Field


class VitalSigns(BaseModel):
    """Objective vital sign measurements recorded during the encounter.

    Each vital is split into a numeric/string value and its unit of measurement.
    """

    temperature: float | None = Field(
        default=None,
        description="Body temperature value, e.g. 100.8.",
    )
    temperature_units: str | None = Field(
        default=None,
        description="Unit for temperature, e.g. 'F' or 'C'.",
    )
    oxygen_saturation: float | None = Field(
        default=None,
        description="Peripheral oxygen saturation (SpO2) value, e.g. 95.",
    )
    oxygen_saturation_units: str | None = Field(
        default=None,
        description="Unit for oxygen saturation, typically '%'.",
    )
    heart_rate: int | None = Field(
        default=None,
        description="Heart rate value, e.g. 72.",
    )
    heart_rate_units: str | None = Field(
        default=None,
        description="Unit for heart rate, typically 'bpm'.",
    )
    respiratory_rate: int | None = Field(
        default=None,
        description="Respiratory rate value, e.g. 16.",
    )
    respiratory_rate_units: str | None = Field(
        default=None,
        description="Unit for respiratory rate, typically 'breaths/min'.",
    )
    blood_pressure: str | None = Field(
        default=None,
        description="Blood pressure value as 'systolic/diastolic', e.g. '120/80'.",
    )
    blood_pressure_units: str | None = Field(
        default=None,
        description="Unit for blood pressure, typically 'mmHg'.",
    )


class HistoryOfPresentIllness(BaseModel):
    """The History of Present Illness (HPI) narrative for the encounter."""

    hpi: str = Field(
        default="",
        description=(
            "A single flowing, chronological narrative paragraph describing the "
            "present illness: chief complaint, onset, duration, intensity, associated "
            "symptoms, exacerbating/alleviating factors, and treatments attempted."
        ),
    )


class PhysicalExamSystem(str, Enum):
    """The body systems that may be documented in the physical exam."""

    GENERAL = "general"
    HEENT = "heent"
    NECK = "neck"
    CARDIOVASCULAR = "cardiovascular"
    RESPIRATORY = "respiratory"
    GASTROINTESTINAL = "gastrointestinal"
    GENITOURINARY = "genitourinary"
    MUSCULOSKELETAL = "musculoskeletal"
    EXTREMITIES = "extremities"
    SKIN = "skin"
    NEUROLOGIC = "neurologic"
    PSYCHIATRIC = "psychiatric"


class PhysicalExamFinding(BaseModel):
    """Documented findings for a single body system examined."""

    system: PhysicalExamSystem = Field(
        description="The body system examined. Must be one of the defined systems.",
    )
    findings: str = Field(
        description="Documented findings for that system, including the absence of expected abnormalities when stated.",
    )


class PhysicalExam(BaseModel):
    """Physical Exam (PE) findings grouped by body system."""

    findings: list[PhysicalExamFinding] = Field(
        default_factory=list,
        description="One entry per body system examined during the encounter.",
    )


class Diagnosis(BaseModel):
    """A single diagnosis with its ICD-10 code.

    Common model reused for both the differential diagnoses and the
    assessment's primary/working diagnosis.
    """

    icd10_code: str = Field(
        description="The ICD-10-CM code for the condition, e.g. 'J18.9'.",
    )
    diagnosis_name: str = Field(
        description="Human-readable name of the condition, e.g. 'Pneumonia, unspecified organism'.",
    )


class DifferentialDiagnosis(Diagnosis):
    """A possible condition considered during the evaluation."""


class Assessment(Diagnosis):
    """The clinician's primary or working diagnosis for the encounter."""


class DiagnosesOutput(BaseModel):
    """Diagnoses portion of a clinical note (differentials + assessment).

    Used as the focused output schema for the diagnoses node so the model isn't
    also responsible for vitals in the same pass. Merged with `VitalSigns` into a
    full `ClinicalNote`.
    """

    differential_diagnoses: list[DifferentialDiagnosis] = Field(
        default_factory=list,
        description="List of possible conditions considered based on the evaluation, including ICD-10 codes.",
    )
    assessment: Assessment = Field(
        description="Primary diagnosis or working diagnosis, including ICD-10 codes.",
    )


class ClinicalNote(BaseModel):
    """Structured clinical note extracted from a patient-doctor transcript."""

    hpi: str = Field(
        default="",
        description="History of Present Illness narrative for the encounter.",
    )
    vital_signs: VitalSigns = Field(
        default_factory=VitalSigns,
        description="Vital signs measured during the visit.",
    )
    physical_exam: list[PhysicalExamFinding] = Field(
        default_factory=list,
        description="Physical exam findings grouped by body system.",
    )
    differential_diagnoses: list[DifferentialDiagnosis] = Field(
        default_factory=list,
        description="List of possible conditions considered based on the evaluation, including ICD-10 codes.",
    )
    assessment: Assessment = Field(
        description="Primary diagnosis or working diagnosis, including ICD-10 codes.",
    )


class HPIJudgeScore(BaseModel):
    """LLM-as-a-Judge assessment of a generated HPI against its source transcript.

    Scores each dimension 0-4; see `prompts/hpi_judge_prompt.txt` for the rubric.
    """

    accuracy_score: int = Field(
        ge=0, le=4, description="Faithfulness of the HPI to the transcript, 0-4."
    )
    accuracy_rationale: str = Field(
        description="2-4 sentences citing the transcript/HPI evidence behind the accuracy score."
    )
    completeness_score: int = Field(
        ge=0,
        le=4,
        description="Coverage of transcript content relevant to the present illness, 0-4.",
    )
    completeness_rationale: str = Field(
        description="2-4 sentences citing the transcript/HPI evidence behind the completeness score."
    )
    tone_score: int = Field(
        ge=0, le=4, description="Physician-documentation register of the HPI, 0-4."
    )
    tone_rationale: str = Field(
        description="2-4 sentences citing the HPI evidence behind the tone score."
    )


class PEJudgeScore(BaseModel):
    """LLM-as-a-Judge assessment of extracted Physical Exam findings against the transcript.

    Scores each dimension 0-4; see `prompts/physical_exam_judge_prompt.txt` for
    the rubric. Covers depth (accuracy/completeness of the findings actually
    included, ignoring placeholder entries) and register (tone); the *breadth*
    of which body systems were included is scored separately, by code, as
    precision/recall against a golden system set.
    """

    accuracy_score: int = Field(
        ge=0, le=4, description="Faithfulness of the non-placeholder PE findings to the transcript's exam, 0-4."
    )
    accuracy_rationale: str = Field(
        description="2-4 sentences citing the transcript/PE evidence behind the accuracy score."
    )
    completeness_score: int = Field(
        ge=0,
        le=4,
        description="Depth of findings captured for each system the exam actually covered, 0-4.",
    )
    completeness_rationale: str = Field(
        description="2-4 sentences citing the transcript/PE evidence behind the completeness score."
    )
    tone_score: int = Field(
        ge=0, le=4, description="PE documentation register of the findings, 0-4."
    )
    tone_rationale: str = Field(
        description="2-4 sentences citing the PE evidence behind the tone score."
    )
