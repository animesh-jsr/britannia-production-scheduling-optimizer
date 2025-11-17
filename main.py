# main.py
import pandas as pd
from ortools.linear_solver import pywraplp

def load_data(path="data/production_data.csv"):
    return pd.read_csv(path)

def build_and_solve_schedule(df):
    products = df["Product"].tolist()
    n = len(products)

    # create solver
    solver = pywraplp.Solver.CreateSolver('SCIP')
    if solver is None:
        raise Exception("Solver not available. Make sure ortools is installed.")

    # decision vars: start and finish times for each product
    start = [solver.NumVar(0, solver.infinity(), f"start_{i}") for i in range(n)]
    finish = [solver.NumVar(0, solver.infinity(), f"finish_{i}") for i in range(n)]

    # process times (bake + cool + pack)
    process_time = {
        products[i]: df.loc[i, "Bake_Time"] + df.loc[i, "Cool_Time"] + df.loc[i, "Pack_Time"]
        for i in range(n)
    }

    changeover = {
        products[i]: df.loc[i, "Changeover_Time"]
        for i in range(n)
    }

    # constraints: finish = start + process_time
    for i, p in enumerate(products):
        solver.Add(finish[i] == start[i] + process_time[p])

    # sequencing constraints: simple linear sequence (you can expand to permutations)
    for i in range(n - 1):
        solver.Add(start[i + 1] >= finish[i] + changeover[products[i]])

    # makespan
    makespan = solver.NumVar(0, solver.infinity(), "makespan")
    solver.Add(makespan >= finish[n - 1])

    # objective: minimize makespan
    solver.Minimize(makespan)

    status = solver.Solve()
    if status != pywraplp.Solver.OPTIMAL:
        print("No optimal solution found. Status:", status)
        return None

    schedule = []
    for i, p in enumerate(products):
        schedule.append({
            "product": p,
            "start": start[i].solution_value(),
            "finish": finish[i].solution_value(),
            "process_time": process_time[p],
            "changeover": changeover[p]
        })

    return {"makespan": makespan.solution_value(), "schedule": schedule}

def print_schedule(result):
    if result is None:
        print("No schedule to print.")
        return
    print("\nOptimal Production Schedule:")
    for item in result["schedule"]:
        print(f"{item['product']}: Start at {item['start']:.2f} min, Finish at {item['finish']:.2f} min (Proc {item['process_time']} min)")
    print(f"\nTotal Production Time (Makespan): {result['makespan']:.2f} min")

if __name__ == "__main__":
    df = load_data("data/production_data.csv")
    result = build_and_solve_schedule(df)
    print_schedule(result)

    # optionally save to results file
    with open("results/schedule_output.txt", "w") as f:
        if result:
            for item in result["schedule"]:
                f.write(f"{item['product']}: {item['start']:.2f} -> {item['finish']:.2f}\n")
            f.write(f"\nMakespan: {result['makespan']:.2f} min\n")
