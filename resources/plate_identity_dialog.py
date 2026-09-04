from __future__ import annotations

import json
from typing import Mapping

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QDialogButtonBox,
    QHeaderView,
    QLabel,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

from .plate_identity import allocate_plate_names, sanitize_plate_id


class PlateIdentityReviewDialog(QDialog):
    def __init__(self, records: list[Mapping[str, object]], parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Review Plate Identities")
        self.resize(1180, 620)
        self._source_records = [dict(record) for record in records]
        self._resolved_records: list[dict[str, object]] = []

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        note = QLabel("Edit uncertain plate IDs before applying. Original image files remain unchanged.")
        note.setStyleSheet("color: #9eb2d8;")
        layout.addWidget(note)

        self.table = QTableWidget(len(self._source_records), 7, self)
        self.table.setHorizontalHeaderLabels(
            ["Original", "Plate ID", "Acquired", "Allocated name", "Confidence", "Status", "Evidence"]
        )
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setAlternatingRowColors(True)
        self.table.verticalHeader().setVisible(False)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeToContents)
        header.setSectionResizeMode(0, QHeaderView.Stretch)
        header.setSectionResizeMode(1, QHeaderView.Stretch)
        header.setSectionResizeMode(3, QHeaderView.Stretch)
        header.setSectionResizeMode(6, QHeaderView.Stretch)
        layout.addWidget(self.table, 1)

        for row, record in enumerate(self._source_records):
            evidence = record.get("evidence", [])
            if isinstance(evidence, list):
                evidence_text = "; ".join(
                    f"{entry.get('kind', 'clue')}: {entry.get('text', '')}"
                    for entry in evidence
                    if isinstance(entry, dict)
                )
            else:
                evidence_text = json.dumps(evidence, ensure_ascii=True)
            values = [
                str(record.get("source_name", "")),
                str(record.get("plate_id", "")),
                str(record.get("source_date", "")),
                str(record.get("allocated_name", "")),
                f"{float(record.get('confidence', 0.0) or 0.0):.2f}",
                str(record.get("status", "")),
                evidence_text,
            ]
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                if column != 1:
                    item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                item.setToolTip(value)
                self.table.setItem(row, column, item)
            status = str(record.get("status", ""))
            if status == "review":
                self.table.item(row, 1).setBackground(QColor(117, 87, 28, 120))
            elif status == "unidentified":
                self.table.item(row, 1).setBackground(QColor(120, 45, 50, 120))

        buttons = QDialogButtonBox(QDialogButtonBox.Apply | QDialogButtonBox.Cancel, self)
        buttons.button(QDialogButtonBox.Apply).setText("Apply Identities")
        buttons.accepted.connect(self._accept_records)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _accept_records(self) -> None:
        edited: list[dict[str, object]] = []
        for row, source in enumerate(self._source_records):
            record = dict(source)
            plate_item = self.table.item(row, 1)
            plate_id = sanitize_plate_id(plate_item.text() if plate_item is not None else "")
            original = sanitize_plate_id(source.get("plate_id", ""))
            record["plate_id"] = plate_id
            if plate_id != original:
                record["user_override"] = plate_id or None
                record["confidence"] = 1.0 if plate_id else 0.0
                record["status"] = "accepted" if plate_id else "unidentified"
            edited.append(record)
        self._resolved_records = allocate_plate_names(edited)
        self.accept()

    def resolved_records(self) -> list[dict[str, object]]:
        return [dict(record) for record in self._resolved_records]
