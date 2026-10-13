import marimo

__generated_with = "0.25.1"
app = marimo.App()


@app.cell
def imports():
    import marimo as mo
    import polars as pl

    return mo, pl


@app.cell
def introduction(mo):
    mo.md("""
    # Choosing when to observe a UI

    A computer-use agent needs to decide how long to wait after an action.
    Adjust the wait budget to inspect which UI changes would arrive later.

    These hand-written rows are synthetic examples, not benchmark results
    or model predictions. All data stays in memory.
    """)
    return


@app.cell
def transition_data(mo, pl):
    transitions = pl.DataFrame(
        {
            "action": [
                "type query",
                "submit search",
                "open menu",
                "open details",
                "navigate back",
                "click disabled button",
            ],
            "before_ui": ["empty form", "filled form", "results", "results", "details", "dialog"],
            "after_ui": ["filled form", "results", "menu", "details", "results", "dialog"],
            "latency_ms": [260, 650, 120, 1100, 900, 50],
        }
    ).with_columns((pl.col("before_ui") != pl.col("after_ui")).alias("state_changed"))
    mo.ui.table(transitions, selection=None)
    return (transitions,)


@app.cell
def wait_control(mo):
    wait_budget = mo.ui.slider(start=0, stop=1500, step=100, value=500, label="Wait budget (ms)")
    wait_budget
    return (wait_budget,)


@app.cell
def reactive_result(mo, pl, transitions, wait_budget):
    budget_ms = wait_budget.value
    over_budget = transitions.filter(
        pl.col("state_changed") & (pl.col("latency_ms") > budget_ms)
    ).sort("latency_ms")
    return (
        budget_ms,
        over_budget,
        mo.vstack(
            [
                mo.md(
                    f"**{over_budget.height} toy UI changes** "
                    f"arrive after the {budget_ms} ms budget."
                ),
                mo.ui.table(over_budget, selection=None),
            ]
        ),
    )


if __name__ == "__main__":
    app.run()
