import csv
import io
import json
import os
import os.path
from collections import defaultdict
from typing import Any, List


class KFPUIMetadata:
    def __init__(self, output_path: str):
        self.output_dir = os.path.dirname(output_path)
        self.output_path = output_path
        os.makedirs(self.output_dir, exist_ok=True)

        self.outputs = []

    def add_confusion_matrix(
        self, truths: List[Any], predicted: List[Any], labels: List[Any]
    ) -> 'KFPUIMetadata':
        self.outputs.append(
            {
                'type': 'confusion_matrix',
                'format': 'csv',
                'schema': [
                    {'name': 'target', 'type': 'CATEGORY'},
                    {'name': 'predicted', 'type': 'CATEGORY'},
                    {'name': 'count', 'type': 'NUMBER'},
                ],
                'source': self.build_confusion_matrix_csv(truths, predicted),
                'storage': 'inline',
                'labels': [str(x) for x in labels],
            }
        )
        return self

    def build_confusion_matrix_csv(
        self, truths: List[Any], predicted: List[Any]
    ) -> str:
        assert len(truths) == len(predicted)
        counts = defaultdict(int)
        for i in range(len(truths)):
            counts[(str(truths[i]), str(predicted[i]))] += 1

        return self._build_csv([[k[0], k[1], v] for k, v in counts.items()])

    def add_markdown(self, content: str) -> 'KFPUIMetadata':
        self.outputs.append(
            {
                'type': 'markdown',
                'storage': 'inline',
                'source': content,
            }
        )
        return self

    def add_roc(
        self, fpr: List[float], tpr: List[float], thresholds: List[float]
    ) -> 'KFPUIMetadata':
        self.outputs.append(
            {
                'type': 'roc',
                'format': 'csv',
                'schema': [
                    {'name': 'fpr', 'type': 'NUMBER'},
                    {'name': 'tpr', 'type': 'NUMBER'},
                    {'name': 'thresholds', 'type': 'NUMBER'},
                ],
                'source': self.build_roc_csv(fpr, tpr, thresholds),
                'storage': 'inline',
            }
        )
        return self

    def build_roc_csv(
        self, fpr: List[float], tpr: List[float], thresholds: List[float]
    ) -> str:
        assert len(fpr) == len(tpr)
        assert len(tpr) == len(thresholds)

        return self._build_csv(
            [['fpr', 'tpr', 'thresholds']]
            + [[fpr[i], tpr[i], thresholds[i]] for i in range(len(thresholds))]
        )

    def add_table(
        self, header: List[str], rows: List[List[Any]]
    ) -> 'KFPUIMetadata':

        self.outputs.append(
            {
                'type': 'table',
                'format': 'csv',
                'header': header,
                'source': self._build_csv(
                    [[str(x) for x in row] for row in rows]
                ),
                'storage': 'inline',
            }
        )

        return self

    def add_tensorboard(self, log_dir: str) -> 'KFPUIMetadata':
        self.outputs.append(
            {
                'type': 'tensorboard',
                'source': log_dir,
            }
        )

        return self

    def add_web_app(self, content: str) -> 'KFPUIMetadata':
        self.outputs.append(
            {
                'type': 'web-app',
                'source': content,
                'storage': 'inline',
            }
        )

        return self

    def _build_csv(self, rows: List[List[Any]]):
        output = io.StringIO()
        writer = csv.writer(output, quoting=csv.QUOTE_NONNUMERIC)
        writer.writerows(rows)
        return output.getvalue()

    def write(self):
        with open(self.output_path, 'w') as fp:
            json.dump({'outputs': self.outputs}, fp)
