"""Bounded, ordered processing tests: no endpoints or timing-ratio assertions."""

import csv
import json
import logging
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import Mock

import pytest

from email_classifier import cli
from email_classifier.classifier import (
    ClassificationResult,
    EmailClassifier,
    HybridClassifier,
    HybridWorkflowLogger,
)
from email_classifier.llm.config import LLMConfig, LLMProvider
from email_classifier.processor import OutputManager, StreamingProcessor


def row(number, **changes):
    result = dict(
        sender="a@example.com",
        receiver="b@example.com",
        subject=str(number),
        body="body",
        timestamp="2024-01-01",
        has_url="true",
        label="1",
    )
    result.update(changes)
    return result


def write_input(tmp_path, rows):
    path = tmp_path / "input.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def read_output(directory, name="email_finance.csv"):
    with (directory / name).open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def details():
    return {
        "method1": {"domain": "finance", "confidence": 0.8},
        "method2": {"domain": "finance", "confidence": 0.9},
        "agreement": True,
    }


class FakeClassifier:
    def __init__(self, calls, action=None):
        self.calls = calls
        self.action = action
        self.owner = None

    def classify_dict(self, email):
        thread = threading.get_ident()
        if self.owner is None:
            self.owner = thread
        assert self.owner == thread
        self.calls.append((email["subject"], thread, id(self)))
        if self.action:
            self.action(email)
        return "finance", details()


@pytest.mark.parametrize("workers", [0, -1, True, False, 1.5, "2", None])
def test_invalid_worker_count(workers):
    with pytest.raises(ValueError, match="integer >= 1"):
        StreamingProcessor(workers=workers)


@pytest.mark.parametrize("value", ["0", "-1", "1.5", "true"])
def test_cli_rejects_workers(monkeypatch, value):
    monkeypatch.setattr(
        "sys.argv", ["email-cli", "classify", "in.csv", "-o", "out", "--workers", value]
    )
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 2


@pytest.mark.parametrize("value", [None, "4"])
@pytest.mark.parametrize("explicit", [False, True])
def test_cli_parses_workers(monkeypatch, value, explicit):
    argv = ["email-cli"] + (["classify"] if explicit else []) + ["in.csv", "-o", "out"]
    if value:
        argv += ["--workers", value]
    monkeypatch.setattr("sys.argv", argv)
    command = Mock(return_value=0)
    monkeypatch.setattr(cli, "cmd_classify", command)
    assert cli.main() == 0
    assert command.call_args.args[0].workers == (int(value) if value else 1)


def test_serial_uses_original_classifier_and_no_factory(tmp_path):
    calls = []
    classifier = FakeClassifier(calls)
    factory = Mock(side_effect=AssertionError("serial must not clone"))
    processor = StreamingProcessor(classifier=classifier, classifier_factory=factory)
    processor.process(write_input(tmp_path, [row(0), row(1)]), tmp_path / "out")
    assert [call[0] for call in calls] == ["0", "1"]
    assert {call[1] for call in calls} == {threading.get_ident()}
    factory.assert_not_called()


def test_overlap_order_bound_and_worker_reuse(tmp_path, monkeypatch):
    first_started, second_finished, release_first = (
        threading.Event() for _ in range(3)
    )
    calls, read_rows, write_threads, callback_threads, instances = [], [], [], [], []
    source_closed = threading.Event()

    def action(email):
        if email["subject"] == "0":
            first_started.set()
            assert release_first.wait(5)
        elif email["subject"] == "1":
            assert first_started.wait(5)
            second_finished.set()

    def factory():
        worker = FakeClassifier(calls, action)
        instances.append(worker)
        return worker

    processor = StreamingProcessor(
        classifier=FakeClassifier([]),
        classifier_factory=factory,
        workers=2,
        chunk_size=1,
    )
    rows = [row(i) for i in range(8)]
    path = write_input(tmp_path, rows)

    def source(_):
        try:
            for email in rows:
                read_rows.append(email["subject"])
                yield email
        finally:
            source_closed.set()

    monkeypatch.setattr(processor, "_stream_emails", source)
    original_write = OutputManager.write_email

    def write(manager, domain, email):
        write_threads.append(threading.get_ident())
        return original_write(manager, domain, email)

    monkeypatch.setattr(OutputManager, "write_email", write)

    def process():
        coordinator = threading.get_ident()
        stats = processor.process(
            path,
            tmp_path / "out",
            include_details=True,
            progress_callback=lambda *_: callback_threads.append(threading.get_ident()),
        )
        return coordinator, stats

    with ThreadPoolExecutor(max_workers=1) as pool:
        task = pool.submit(process)
        try:
            assert first_started.wait(5)
            assert second_finished.wait(5)  # Actual overlap, no speed-ratio assumption.
            assert read_rows == ["0", "1"]  # Finished second row still occupies window.
            assert not write_threads
        finally:
            release_first.set()
        coordinator, stats = task.result(timeout=5)
    assert [email["subject"] for email in read_output(tmp_path / "out")] == list(
        map(str, range(8))
    )
    assert stats.total_processed == 8
    assert set(write_threads + callback_threads) == {coordinator}
    assert all(thread != coordinator for _, thread, _ in calls)
    assert len(instances) == 2
    assert any(
        count > 1 for count in Counter(instance for _, _, instance in calls).values()
    )
    assert len({worker.owner for worker in instances}) == 2
    assert source_closed.is_set()


class FakeHybrid(HybridClassifier):
    def __init__(self, calls, action=None):
        super().__init__()
        self.calls = calls
        self.action = action

    def classify_dict(self, email, email_idx=0, total_emails=0):
        self.calls.append((email["subject"], email_idx))
        assert self.status_callback is None
        self.stats.total_processed += 1
        self.stats.llm_call_count += 1
        self.stats.llm_total_time_ms += 12
        self.stats.total_processing_time_ms += 20
        self.stats.llm_rejected_count += 1
        self.workflow_logger.log_step(
            email_idx, "start", extra={"subject": email["subject"]}
        )
        if self.action:
            self.action(email)
        self.workflow_logger.log_step(email_idx, "final_result", result="finance")
        return "finance", details()


def test_hybrid_order_indices_skips_partial_failure_and_stats(tmp_path, monkeypatch):
    calls, replay_threads, callback_threads = [], [], []
    completed, release = threading.Event(), threading.Event()

    def action(email):
        subject = email["subject"]
        if subject == "0":
            assert release.wait(5)
        elif subject == "3":
            completed.set()
        elif subject == "4":
            raise RuntimeError("failed once")

    logger = HybridWorkflowLogger(str(tmp_path / "workflow.jsonl"))
    coordinator = HybridClassifier(
        workflow_logger=logger,
        status_callback=lambda *_: callback_threads.append(threading.get_ident()),
    )
    emit = logger.emit_entry

    def replay(entry):
        replay_threads.append(threading.get_ident())
        emit(entry)

    monkeypatch.setattr(logger, "emit_entry", replay)
    processor = StreamingProcessor(
        classifier=coordinator,
        use_hybrid=True,
        workers=4,
        classifier_factory=lambda: FakeHybrid(calls, action),
        max_body_length=10,
    )
    path = write_input(
        tmp_path,
        [row(0), row(1, sender="invalid"), row(2, body="x" * 11), row(3), row(4)],
    )
    with ThreadPoolExecutor(max_workers=1) as pool:

        def process():
            return threading.get_ident(), processor.process(path, tmp_path / "out")

        task = pool.submit(process)
        try:
            assert completed.wait(5)
        finally:
            release.set()
        thread, stats = task.result(timeout=5)
    logger.close()
    entries = [
        json.loads(line)
        for line in (tmp_path / "workflow.jsonl").read_text().splitlines()
    ]
    assert [entry["email_idx"] for entry in entries] == [0, 0, 3, 3, 4]
    assert sorted(calls) == [("0", 0), ("3", 3), ("4", 4)]
    assert set(replay_threads + callback_threads) == {thread}
    assert [email["subject"] for email in read_output(tmp_path / "out")] == ["0", "3"]
    assert [
        email["subject"] for email in read_output(tmp_path / "out", "email_unsure.csv")
    ] == ["4"]
    assert stats.errors == 1
    assert stats.validation_stats.total_invalid == 1
    assert stats.skipped_stats.total_skipped == 1
    assert stats.hybrid_workflow.llm_call_count == 3
    assert stats.hybrid_workflow.llm_rejected_count == 3
    assert stats.hybrid_workflow.llm_total_time_ms == 36
    assert coordinator.stats.total_processing_time_ms == 60


def test_parallel_hybrid_status_failure_does_not_duplicate_committed_rows(
    tmp_path, caplog
):
    calls, notifications = [], []

    def status(message):
        notifications.append(message)
        raise RuntimeError("notification failed")

    coordinator = HybridClassifier(status_callback=status)
    processor = StreamingProcessor(
        classifier=coordinator,
        use_hybrid=True,
        workers=2,
        classifier_factory=lambda: FakeHybrid(calls),
    )
    stats = processor.process(
        write_input(tmp_path, [row(i) for i in range(3)]), tmp_path / "out"
    )
    records = [
        email
        for path in (tmp_path / "out").glob("email_*.csv")
        for email in read_output(tmp_path / "out", path.name)
    ]
    assert Counter(email["subject"] for email in records) == {"0": 1, "1": 1, "2": 1}
    assert all(email["classified_domain"] == "finance" for email in records)
    assert stats.total_processed == stats.total_classified == 3
    assert stats.total_unsure == stats.errors == 0
    assert sum(stats.domain_counts.values()) == stats.domain_counts["finance"] == 3
    assert (
        stats.method_agreement_count
        == stats.hybrid_workflow.total_hybrid_processed
        == 3
    )
    assert len(calls) == len(notifications) == 3
    assert caplog.text.count("notification failed") == 3


@pytest.mark.parametrize("interruption", [KeyboardInterrupt, SystemExit])
def test_parallel_hybrid_status_interruption_propagates(tmp_path, interruption):
    def status(_):
        raise interruption("stop notification")

    processor = StreamingProcessor(
        classifier=HybridClassifier(status_callback=status),
        use_hybrid=True,
        workers=2,
        classifier_factory=lambda: FakeHybrid([]),
    )
    with pytest.raises(interruption, match="stop notification"):
        processor.process(write_input(tmp_path, [row(0), row(1)]), tmp_path / "out")
    assert [email["subject"] for email in read_output(tmp_path / "out")] == ["0"]
    assert processor.stats.total_processed == processor.stats.total_classified == 1
    assert processor.stats.total_unsure == processor.stats.errors == 0


@pytest.mark.parametrize(
    "error,unsure",
    [(ValueError("bad"), 0), (RuntimeError("bad"), 1), (TimeoutError("timeout"), 1)],
)
def test_worker_errors_keep_serial_semantics_no_retry(tmp_path, error, unsure):
    calls = []

    def action(_):
        raise error

    processor = StreamingProcessor(
        classifier=FakeClassifier([]),
        workers=2,
        classifier_factory=lambda: FakeClassifier(calls, action),
    )
    stats = processor.process(write_input(tmp_path, [row(0)]), tmp_path / "out")
    assert len(calls) == 1
    assert stats.errors == 1
    assert stats.total_unsure == unsure
    assert stats.total_processed == 0


@pytest.mark.parametrize("abort", ["strict", "interrupt", "write"])
def test_abort_closes_source_executor_and_outputs(tmp_path, monkeypatch, abort):
    closed, started, release = (threading.Event() for _ in range(3))
    calls, instances = [], []
    rows = [row(i) for i in range(8)]
    if abort == "strict":
        rows[0]["sender"] = "invalid"

    def action(email):
        if email["subject"] == "1":
            started.set()
            assert release.wait(5)
        if abort == "interrupt" and email["subject"] == "0":
            assert started.wait(5)
            raise KeyboardInterrupt()

    def factory():
        worker = FakeClassifier(calls, action)
        instances.append(worker)
        return worker

    processor = StreamingProcessor(
        classifier=FakeClassifier([]),
        workers=2,
        strict_validation=abort == "strict",
        classifier_factory=factory,
    )

    def source(_):
        try:
            yield from rows
        finally:
            closed.set()

    monkeypatch.setattr(processor, "_stream_emails", source)
    if abort == "write":

        def write(*_):
            assert started.wait(5)
            raise KeyboardInterrupt()

        monkeypatch.setattr(OutputManager, "write_email", write)
    managers = []
    close = OutputManager.close_all

    def close_all(manager):
        close(manager)
        managers.append(manager)

    monkeypatch.setattr(OutputManager, "close_all", close_all)
    with ThreadPoolExecutor(max_workers=1) as pool:
        task = pool.submit(
            processor.process, write_input(tmp_path, rows), tmp_path / "out"
        )
        try:
            # Strict validation may cancel row 1 before it starts; other cases require overlap.
            if abort != "strict":
                assert started.wait(5)
        finally:
            release.set()
        with pytest.raises(ValueError if abort == "strict" else KeyboardInterrupt):
            task.result(timeout=5)
    assert closed.is_set()
    assert managers and not managers[0].files
    assert all(int(subject) < 2 for subject, _, _ in calls)
    assert not any(t.name.startswith("email_") for t in threading.enumerate())


@pytest.mark.parametrize("interruption", [KeyboardInterrupt, SystemExit])
def test_source_interruption_cancels_queue_without_committing(
    tmp_path, monkeypatch, interruption
):
    started, release, cancelled, closed = (threading.Event() for _ in range(4))
    calls, futures, writes = [], [], []

    class SingleExecutor(ThreadPoolExecutor):
        def __init__(self, **kwargs):
            super().__init__(
                max_workers=1, thread_name_prefix=kwargs["thread_name_prefix"]
            )

        def submit(self, *args, **kwargs):
            future = super().submit(*args, **kwargs)
            futures.append(future)

            def done(task):
                if task.cancelled():
                    cancelled.set()

            future.add_done_callback(done)
            return future

    monkeypatch.setattr("email_classifier.processor.ThreadPoolExecutor", SingleExecutor)

    def action(_):
        started.set()
        assert release.wait(5)

    processor = StreamingProcessor(
        workers=3, classifier_factory=lambda: FakeClassifier(calls, action)
    )

    def source(_):
        try:
            yield row(0)
            assert started.wait(5)
            yield row(1)
            raise interruption("source interrupted")
        finally:
            closed.set()

    monkeypatch.setattr(processor, "_stream_emails", source)
    write = OutputManager.write_email

    def record_write(manager, domain, email):
        writes.append(email["subject"])
        return write(manager, domain, email)

    monkeypatch.setattr(OutputManager, "write_email", record_write)
    with ThreadPoolExecutor(max_workers=1) as pool:
        task = pool.submit(
            processor.process, write_input(tmp_path, [row(0), row(1)]), tmp_path / "out"
        )
        try:
            assert cancelled.wait(5)
            assert len(futures) == 2
            assert futures[1].cancelled()
            assert not futures[0].done()
            assert not task.done()  # Shutdown waits for the blocked running task.
            assert not writes
        finally:
            release.set()
        with pytest.raises(interruption, match="source interrupted"):
            task.result(timeout=5)
    assert closed.is_set()
    assert [subject for subject, _, _ in calls] == ["0"]
    assert not writes
    assert processor.stats.total_processed == processor.stats.total_unsure == 0
    assert processor.stats.total_classified == processor.stats.errors == 0
    assert all(
        not read_output(tmp_path / "out", path.name)
        for path in (tmp_path / "out").glob("email_*.csv")
    )
    assert not any(t.name.startswith("email_") for t in threading.enumerate())


def test_source_error_is_deferred_until_prior_rows_commit(tmp_path, monkeypatch):
    processor = StreamingProcessor(workers=3)

    def source(_):
        yield row(0)
        yield row(1)
        raise OSError("source failed")

    monkeypatch.setattr(processor, "_stream_emails", source)
    with pytest.raises(OSError, match="source failed"):
        processor.process(write_input(tmp_path, [row(0)]), tmp_path / "out")
    assert processor.stats.total_processed == 2


@pytest.mark.parametrize("custom", ["subclass", "injected", "patched"])
def test_custom_classifier_requires_factory(tmp_path, custom):
    if custom == "subclass":

        class Custom(EmailClassifier):
            pass

        classifier = Custom()
    else:
        classifier = EmailClassifier()
        if custom == "injected":
            classifier.method1 = Mock()
        else:
            classifier.classify_dict = Mock()
    processor = StreamingProcessor(classifier=classifier, workers=2)
    with pytest.raises(ValueError, match="classifier_factory"):
        processor.process(write_input(tmp_path, [row(0)]), tmp_path / "out")


@pytest.mark.parametrize("hybrid", [False, True])
def test_builtin_clones_snapshot_configuration_and_isolate_clients(monkeypatch, hybrid):
    clients = []

    def create(config):
        client = Mock(config=config)
        clients.append(client)
        return client

    monkeypatch.setattr("email_classifier.llm.create_classifier", create)
    config = LLMConfig(provider=LLMProvider.OLLAMA, model="snapshot", timeout=17)
    template = (
        HybridClassifier(llm_config=config, llm_confidence_cutoff=0.7)
        if hybrid
        else EmailClassifier(llm_config=config)
    )
    processor = StreamingProcessor(classifier=template, workers=2, use_hybrid=hybrid)
    factory = processor._worker_factory()
    config.model = "mutated"
    monkeypatch.setattr(
        LLMConfig, "from_env", Mock(side_effect=AssertionError("no environment reload"))
    )
    first, second = factory(), factory()
    assert len({id(client) for client in clients}) == 3
    assert first._llm_config.model == second._llm_config.model == "snapshot"
    assert first._llm_config.timeout == 17
    assert first._llm_config is not second._llm_config
    assert first.domains == second.domains == template.domains
    assert first.domains is not second.domains
    if hybrid:
        assert first.llm_confidence_cutoff == 0.7
        assert first.status_callback is None
        assert first.workflow_logger is None
    else:
        assert first.weight_method_3 == template.weight_method_3


@pytest.mark.parametrize("mode", ["classic", "hybrid", "force"])
def test_real_classifiers_serial_parallel_equivalence_with_fake_llm(
    tmp_path, monkeypatch, mode
):
    llm_calls = []

    def create(_):
        def classify(email):
            llm_calls.append(email.subject)
            if email.subject == "failure":
                raise RuntimeError("mocked provider failure")
            return ClassificationResult("retail", 0.1, {"retail": 0.1}, "typesafe")

        class FakeLLM:
            def classify(self, email):
                return classify(email)

        return FakeLLM()

    monkeypatch.setattr("email_classifier.llm.create_classifier", create)
    # Deterministic timings let us compare merged durations as well as counters.
    monkeypatch.setattr("email_classifier.classifier.time.perf_counter", lambda: 1.0)
    config = (
        None
        if mode == "classic"
        else LLMConfig(provider=LLMProvider.OLLAMA, model="mock")
    )

    def classifier():
        return (
            HybridClassifier(llm_config=config)
            if mode == "hybrid"
            else EmailClassifier(llm_config=config)
        )

    path = write_input(
        tmp_path, [row(0, body="invoice payment bank account"), row("failure"), row(2)]
    )
    stats = []
    for workers in (1, 3):
        processor = StreamingProcessor(
            classifier=classifier(), use_hybrid=mode == "hybrid", workers=workers
        )
        stats.append(
            processor.process(
                path, tmp_path / str(workers), include_details=True
            ).to_dict()
        )
    for stat in stats:
        for key in ("start_time", "end_time", "duration_seconds"):
            stat.pop(key)
    assert stats[0] == stats[1]
    names = {p.name for p in (tmp_path / "1").glob("*.csv")}
    assert names == {p.name for p in (tmp_path / "3").glob("*.csv")}
    for name in names:
        assert (tmp_path / "1" / name).read_text() == (
            tmp_path / "3" / name
        ).read_text()
    if mode == "hybrid":
        assert stats[1]["hybrid_workflow"]["llm_rejected_count"] > 0
    if mode == "force":
        assert Counter(llm_calls) == {"0": 2, "failure": 2, "2": 2}


def test_factory_failure_is_a_single_row_error(tmp_path):
    factory = Mock(side_effect=RuntimeError("factory failed"))
    processor = StreamingProcessor(workers=2, classifier_factory=factory)
    stats = processor.process(write_input(tmp_path, [row(0)]), tmp_path / "out")
    assert stats.errors == stats.total_unsure == 1
    factory.assert_called_once()


def test_all_source_rows_occupy_window_even_without_tasks(tmp_path, monkeypatch):
    started, release, filled_window = (threading.Event() for _ in range(3))
    calls, read_rows = [], []
    rows = [row(0), row(1, sender="invalid"), row(2, body="x" * 11), row(3)]

    def action(email):
        if email["subject"] == "0":
            started.set()
            assert release.wait(5)

    processor = StreamingProcessor(
        workers=3,
        max_body_length=10,
        classifier_factory=lambda: FakeClassifier(calls, action),
    )

    def source(_):
        for email in rows:
            read_rows.append(email["subject"])
            yield email

    monkeypatch.setattr(processor, "_stream_emails", source)
    schedule = processor._parallel_rows

    def scheduled(*args):
        iterator = schedule(*args)
        try:
            for item in iterator:
                filled_window.set()
                yield item
        finally:
            iterator.close()

    monkeypatch.setattr(processor, "_parallel_rows", scheduled)
    with ThreadPoolExecutor(max_workers=1) as pool:
        task = pool.submit(
            processor.process, write_input(tmp_path, rows), tmp_path / "out"
        )
        try:
            assert started.wait(5)
            assert filled_window.wait(5)
            assert read_rows == ["0", "1", "2"]
        finally:
            release.set()
        stats = task.result(timeout=5)
    assert [subject for subject, _, _ in calls] == ["0", "3"]
    assert (
        stats.validation_stats.total_invalid == stats.skipped_stats.total_skipped == 1
    )


def test_output_failure_does_not_repeat_classification(tmp_path, monkeypatch):
    calls = []
    write = OutputManager.write_email

    def failing_write(manager, domain, email):
        if domain == "finance" and email["subject"] == "0":
            raise OSError("write failed")
        return write(manager, domain, email)

    monkeypatch.setattr(OutputManager, "write_email", failing_write)
    processor = StreamingProcessor(
        workers=2, classifier_factory=lambda: FakeClassifier(calls)
    )
    stats = processor.process(write_input(tmp_path, [row(0), row(1)]), tmp_path / "out")
    assert Counter(subject for subject, _, _ in calls) == {"0": 1, "1": 1}
    assert stats.errors == stats.total_unsure == stats.total_processed == 1
    assert [
        email["subject"] for email in read_output(tmp_path / "out", "email_unsure.csv")
    ] == ["0"]


@pytest.mark.parametrize("hybrid", [False, True])
def test_manually_injected_live_client_requires_factory(tmp_path, hybrid):
    config = LLMConfig(
        provider=LLMProvider.TYPESAFE,
        model="mock",
        typesafe_base_url="http://mock.invalid/v1",
    )
    template = (
        HybridClassifier(llm_config=config)
        if hybrid
        else EmailClassifier(llm_config=config)
    )
    method = template.llm_classifier if hybrid else template.method3
    method._client = Mock()
    processor = StreamingProcessor(classifier=template, workers=2)
    with pytest.raises(ValueError, match="initialized clients.*classifier_factory"):
        processor.process(write_input(tmp_path, [row(0)]), tmp_path / "out")


def test_application_diagnostics_replay_in_order_on_coordinator(tmp_path, monkeypatch):
    logger = logging.getLogger("email_classifier.classifier")
    emitted, calls = [], []
    original_filters = list(logger.filters)
    completed, release = threading.Event(), threading.Event()

    class Handler(logging.Handler):
        def emit(self, record):
            emitted.append((record.getMessage(), threading.get_ident()))

    handler = Handler()
    logger.addHandler(handler)
    original_level = logger.level
    logger.setLevel(logging.INFO)

    def factory():
        logger.info("worker initialized")

        def action(email):
            if email["subject"] == "0":
                assert release.wait(5)
            else:
                completed.set()
            logger.warning("diagnostic %s", email["subject"])

        return FakeClassifier(calls, action)

    processor = StreamingProcessor(workers=2, classifier_factory=factory)
    try:
        with ThreadPoolExecutor(max_workers=1) as pool:

            def process():
                return threading.get_ident(), processor.process(
                    write_input(tmp_path, [row(0), row(1)]), tmp_path / "out"
                )

            task = pool.submit(process)
            try:
                assert completed.wait(5)
                assert emitted == []
            finally:
                release.set()
            coordinator, _ = task.result(timeout=5)
        assert [message for message, _ in emitted] == [
            "worker initialized",
            "diagnostic 0",
            "worker initialized",
            "diagnostic 1",
        ]
        assert {thread for _, thread in emitted} == {coordinator}
        assert logger.filters == original_filters
    finally:
        logger.removeHandler(handler)
        logger.setLevel(original_level)


def test_cli_passes_workers_and_displays_configuration(tmp_path, monkeypatch):
    from argparse import Namespace

    ui = Mock()
    monkeypatch.setattr(cli, "get_ui", lambda **_: ui)
    monkeypatch.setattr(cli, "verify_prerequisites", lambda **_: (True, [], None))
    monkeypatch.setattr(cli, "setup_logging", lambda *_, **__: Mock())
    monkeypatch.setattr(cli, "RICH_AVAILABLE", False)
    processor = Mock()
    constructor = Mock(return_value=processor)
    monkeypatch.setattr(cli, "StreamingProcessor", constructor)
    args = Namespace(
        input=str(tmp_path / "in.csv"),
        output=str(tmp_path / "out"),
        quiet=False,
        use_llm=False,
        log_file=None,
        verbose=False,
        workers=4,
        chunk_size=7,
        include_details=False,
        strict_validation=False,
        max_body_length=None,
        allow_large_fields=True,
        no_report=True,
    )
    assert cli.cmd_classify(args) == 0
    assert constructor.call_args.kwargs["workers"] == 4
    assert ui.print_config.call_args.args[2]["Workers"] == 4


def test_early_close_cancels_queued_tasks_and_discards_diagnostics(
    tmp_path, monkeypatch, caplog
):
    started, release, cancelled, closed = (threading.Event() for _ in range(4))
    calls, futures = [], []
    logger = logging.getLogger("email_classifier.classifier")

    class SingleExecutor(ThreadPoolExecutor):
        def __init__(self, **kwargs):
            # Deliberately keep tasks queued to exercise cancellation deterministically.
            super().__init__(
                max_workers=1, thread_name_prefix=kwargs["thread_name_prefix"]
            )

        def submit(self, *args, **kwargs):
            future = super().submit(*args, **kwargs)
            futures.append(future)

            def done(_):
                if sum(task.cancelled() for task in futures) == 2:
                    cancelled.set()

            future.add_done_callback(done)
            return future

    monkeypatch.setattr("email_classifier.processor.ThreadPoolExecutor", SingleExecutor)
    processor = StreamingProcessor(workers=3)

    def source(_):
        try:
            yield from [row(0), row(1), row(2)]
        finally:
            closed.set()

    monkeypatch.setattr(processor, "_stream_emails", source)

    def action(_):
        logger.warning("abandoned diagnostic")
        started.set()
        assert release.wait(5)

    iterator = processor._parallel_rows(
        tmp_path / "unused.csv", 3, lambda: FakeClassifier(calls, action)
    )
    next(iterator)
    assert started.wait(5)
    assert len(futures) == 3
    with ThreadPoolExecutor(max_workers=1) as pool:
        closing = pool.submit(iterator.close)
        try:
            assert cancelled.wait(5)
            assert (
                not closing.done()
            )  # Running request must finish before shutdown returns.
        finally:
            release.set()
        closing.result(timeout=5)
    assert [subject for subject, _, _ in calls] == ["0"]
    assert closed.is_set()
    assert "abandoned diagnostic" not in caplog.text


def test_hybrid_merged_counters_match_serial_all_paths(tmp_path, monkeypatch):
    monkeypatch.setattr("email_classifier.classifier.time.perf_counter", lambda: 1.0)
    llm_calls = []

    class Classic:
        def __init__(self, structural=False):
            self.structural = structural

        def classify(self, email):
            domain = (
                "finance"
                if not self.structural or email.subject == "agree"
                else "technology"
            )
            return ClassificationResult(domain, 0.8, {domain: 0.8}, "classic")

    class LLM:
        def classify(self, email):
            llm_calls.append(email.subject)
            if email.subject == "failure":
                raise TimeoutError("mocked timeout")
            confidence = 0.1 if email.subject == "reject" else 0.9
            return ClassificationResult(
                "retail", confidence, {"retail": confidence}, "typesafe"
            )

    def factory():
        classifier = HybridClassifier(llm_confidence_cutoff=0.5)
        classifier.method1 = Classic()
        classifier.method2 = Classic(structural=True)
        classifier.llm_classifier = LLM()
        return classifier

    path = write_input(
        tmp_path, [row("agree"), row("reject"), row("failure"), row("accept")]
    )
    summaries = []
    for workers in (1, 3):
        processor = StreamingProcessor(
            classifier=factory(),
            use_hybrid=True,
            workers=workers,
            classifier_factory=factory,
        )
        summaries.append(
            processor.process(path, tmp_path / str(workers)).hybrid_workflow.to_dict()
        )
    assert summaries[0] == summaries[1]
    assert summaries[1]["total_hybrid_processed"] == 4
    assert summaries[1]["classic_agreement_count"] == 1
    assert summaries[1]["llm_call_count"] == 2  # Serial excludes calls that raise.
    assert summaries[1]["llm_rejected_count"] == 1
    assert Counter(llm_calls) == {"reject": 2, "failure": 2, "accept": 2}
