"""Dataset and local experiment helpers for the Langfuse SDK."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from langfuse import Langfuse
from langfuse.api import Dataset, DatasetItem, DatasetStatus
from langfuse.experiment import EvaluatorFunction, ExperimentResult, LocalExperimentItem, RunEvaluatorFunction, TaskFunction

from digintlab_langfuse.types import JsonValue, to_json_value


@dataclass(frozen=True, slots=True)
class DatasetConfig:
    """Configuration for creating a Langfuse dataset."""

    name: str
    description: str | None = None
    metadata: dict[str, JsonValue] | None = None
    input_schema: JsonValue = None
    expected_output_schema: JsonValue = None


type ExperimentItem = LocalExperimentItem


def create_dataset(lf: Langfuse, config: DatasetConfig) -> Dataset:
    """Create a dataset in Langfuse."""
    return lf.create_dataset(
        name=config.name,
        description=config.description,
        metadata=config.metadata,
        input_schema=config.input_schema,
        expected_output_schema=config.expected_output_schema,
    )


def create_dataset_item(
    lf: Langfuse,
    *,
    dataset_name: str,
    input: JsonValue = None,  # noqa: A002
    expected_output: JsonValue = None,
    metadata: dict[str, JsonValue] | None = None,
    source_trace_id: str | None = None,
    source_observation_id: str | None = None,
    status: DatasetStatus | None = None,
    id: str | None = None,  # noqa: A002
) -> DatasetItem:
    """Create or upsert one dataset item in Langfuse."""
    return lf.create_dataset_item(
        dataset_name=dataset_name,
        input=input,
        expected_output=expected_output,
        metadata=metadata,
        source_trace_id=source_trace_id,
        source_observation_id=source_observation_id,
        status=status,
        id=id,
    )


def run_local_experiment(
    lf: Langfuse,
    *,
    name: str,
    data: Sequence[ExperimentItem],
    task: TaskFunction,
    run_name: str | None = None,
    description: str | None = None,
    evaluators: Sequence[EvaluatorFunction] = (),
    run_evaluators: Sequence[RunEvaluatorFunction] = (),
    max_concurrency: int = 50,
    metadata: dict[str, str] | None = None,
) -> ExperimentResult:
    """Run a Langfuse experiment over local in-memory items."""
    normalized_data: list[LocalExperimentItem] = []
    for item in data:
        normalized_item: LocalExperimentItem = {}
        if "input" in item:
            normalized_item["input"] = to_json_value(item["input"])
        if "expected_output" in item:
            normalized_item["expected_output"] = to_json_value(item["expected_output"])
        if "metadata" in item:
            metadata_value = item["metadata"]
            match metadata_value:
                case None:
                    normalized_item["metadata"] = None
                case Mapping():
                    normalized_item["metadata"] = {str(key): to_json_value(value) for key, value in metadata_value.items()}
                case _:
                    msg = "Experiment item metadata must be a mapping or None."
                    raise TypeError(msg)
        normalized_data.append(normalized_item)
    return lf.run_experiment(
        name=name,
        run_name=run_name,
        description=description,
        data=normalized_data,
        task=task,
        evaluators=list(evaluators),
        run_evaluators=list(run_evaluators),
        max_concurrency=max_concurrency,
        metadata=metadata,
    )
