from abc import ABC, abstractmethod
from typing import Any, Optional


class PipelineStep(ABC):

    def __init__(self, name: Optional[str] = None):
        self.name = name or self.__class__.__name__

    @abstractmethod
    def process(self, data: Any) -> Any:
        pass

    def __call__(self, data: Any) -> Any:
        print(f"Executing step: {self.name}")
        return self.process(data)


class Pipeline(PipelineStep):

    def __init__(self, name: str = "Pipeline"):
        super().__init__(name=name)
        self.steps: list[PipelineStep] = []

    def add_step(self, step: PipelineStep) -> 'Pipeline':
        self.steps.append(step)
        return self

    def process(self, data: Any) -> Any:
        _print_section(f"Starting Pipeline: {self.name}")

        result = data
        for step in self.steps:
            result = step(result)

        _print_section(f"Pipeline '{self.name}' completed successfully")
        return result


class BranchingPipeline(PipelineStep):

    def __init__(self, name: str = "BranchingPipeline", branches: Optional[dict[str, PipelineStep]] = None):
        super().__init__(name=name)
        self.branches: dict[str, PipelineStep] = branches if branches is not None else {}

    def process(self, data: Any) -> dict[str, Any]:
        _print_section(f"Starting Branching Pipeline: {self.name}")

        results = {}

        for branch_name in self.branches.keys():
            results[branch_name] = self.branches[branch_name](data)

        _print_section(f"Branching Pipeline '{self.name}' completed successfully\nExecuted {len(results)} branch(es): {', '.join(results.keys())}")
        return results


def _print_section(message: str) -> None:
    print(f"\n{'=' * 80}")
    print(message)
    print(f"{'=' * 80}")
