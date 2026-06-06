"""
Food suggestion optimizer using linear programming.
Finds a combination of foods that best meets daily nutritional targets.
"""

import numpy as np
from scipy.optimize import linprog, minimize
from dataclasses import dataclass


@dataclass
class Food:
    name: str
    calories: float      # kcal per 100g
    protein: float       # g per 100g
    carbs: float         # g per 100g
    fat: float           # g per 100g
    fiber: float         # g per 100g
    cost_per_100g: float # EUR per 100g


# Food database (per 100g)
FOODS = [
    Food("Chicken breast",  165, 31.0,  0.0, 3.6, 0.0, 0.25),
    Food("Brown rice",      216,  4.5, 45.0, 1.8, 1.8, 0.08),
    Food("Broccoli",         34,  2.8,  6.6, 0.4, 2.6, 0.12),
    Food("Eggs",            155, 13.0,  1.1, 11.0, 0.0, 0.20),
    Food("Salmon",          208, 20.0,  0.0, 13.0, 0.0, 0.60),
    Food("Sweet potato",     86,  1.6, 20.1, 0.1, 3.0, 0.10),
    Food("Greek yogurt",     59,  10.0, 3.6, 0.4, 0.0, 0.18),
    Food("Lentils",         116,  9.0, 20.0, 0.4, 7.9, 0.09),
    Food("Spinach",          23,  2.9,  3.6, 0.4, 2.2, 0.15),
    Food("Oats",            389, 17.0, 66.0, 7.0, 10.5, 0.07),
    Food("Banana",           89,  1.1, 23.0, 0.3, 2.6, 0.05),
    Food("Almonds",         579, 21.0, 22.0, 50.0, 12.5, 0.40),
    Food("Pasta",           371, 13.0, 75.0, 1.5, 3.2, 0.07),
    Food("Tuna (canned)",   116, 26.0,  0.0, 1.0, 0.0, 0.20),
    Food("Chickpeas",       164,  8.9, 27.4, 2.6, 7.6, 0.10),
]

# Daily nutritional targets (for an average adult)
TARGETS = {
    "calories": 2000,   # kcal
    "protein":    50,   # g
    "carbs":     250,   # g
    "fat":        65,   # g
    "fiber":      25,   # g
}

# Portion constraints per food item (in 100g units): [min, max]
PORTION_BOUNDS = (0.0, 8.0)  # 0g to 800g per food


def build_nutrition_matrix(foods: list[Food]) -> np.ndarray:
    """Returns matrix of shape (n_foods, 5) with nutritional values per 100g."""
    return np.array([
        [f.calories, f.protein, f.carbs, f.fat, f.fiber]
        for f in foods
    ])


def optimize_meal(
    foods: list[Food] = FOODS,
    targets: dict = TARGETS,
    max_foods: int = 5,
    verbose: bool = True,
) -> dict:
    """
    Optimize food portions to minimize deviation from nutritional targets.
    Uses L2 penalty on relative deviation from each target.
    Returns selected foods with portions and achieved nutrition.
    """
    nutrition = build_nutrition_matrix(foods)
    target_vec = np.array([
        targets["calories"], targets["protein"],
        targets["carbs"], targets["fat"], targets["fiber"]
    ], dtype=float)
    weights = 1.0 / target_vec  # normalize so all targets are on equal footing

    def objective(portions: np.ndarray) -> float:
        achieved = nutrition.T @ portions  # shape (5,)
        deviation = (achieved - target_vec) * weights
        return float(np.dot(deviation, deviation))

    def objective_grad(portions: np.ndarray) -> np.ndarray:
        achieved = nutrition.T @ portions
        deviation = (achieved - target_vec) * weights
        return 2.0 * nutrition @ (deviation * weights)

    n = len(foods)
    bounds = [PORTION_BOUNDS] * n
    # Start with portions that roughly hit calorie target spread evenly
    avg_calories = np.mean([f.calories for f in foods])
    x0 = np.ones(n) * (target_vec[0] / (n * avg_calories))

    result = minimize(
        objective,
        x0,
        jac=objective_grad,
        method="L-BFGS-B",
        bounds=bounds,
    )

    portions = result.x
    # Keep only top-max_foods by portion size
    top_indices = np.argsort(portions)[-max_foods:]
    mask = np.zeros(n, dtype=bool)
    mask[top_indices] = True

    selected_portions = portions * mask
    achieved = nutrition.T @ selected_portions

    meal = {
        "foods": [
            {
                "name": foods[i].name,
                "portion_g": round(selected_portions[i] * 100),
            }
            for i in range(n) if mask[i]
        ],
        "nutrition": {
            "calories": round(achieved[0]),
            "protein_g": round(achieved[1], 1),
            "carbs_g":   round(achieved[2], 1),
            "fat_g":     round(achieved[3], 1),
            "fiber_g":   round(achieved[4], 1),
        },
        "targets": targets,
        "score": round(result.fun, 4),
    }

    if verbose:
        _print_meal(meal)

    return meal


def _print_meal(meal: dict) -> None:
    print("\n" + "=" * 45)
    print("  Optimized Meal Suggestion")
    print("=" * 45)
    print(f"{'Food':<20} {'Portion':>10}")
    print("-" * 32)
    for item in meal["foods"]:
        print(f"  {item['name']:<18} {item['portion_g']:>6}g")

    print("\n  Achieved Nutrition  |  Target")
    print("-" * 45)
    n = meal["nutrition"]
    t = meal["targets"]
    print(f"  Calories: {n['calories']:>5} kcal  |  {t['calories']} kcal")
    print(f"  Protein:  {n['protein_g']:>5} g     |  {t['protein']} g")
    print(f"  Carbs:    {n['carbs_g']:>5} g     |  {t['carbs']} g")
    print(f"  Fat:      {n['fat_g']:>5} g     |  {t['fat']} g")
    print(f"  Fiber:    {n['fiber_g']:>5} g     |  {t['fiber']} g")
    print(f"\n  Optimization score (lower = better): {meal['score']}")
    print("=" * 45)


if __name__ == "__main__":
    print("Optimizing meal for default nutritional targets...")
    optimize_meal()

    print("\nOptimizing high-protein meal (80g protein target)...")
    high_protein_targets = {**TARGETS, "protein": 80}
    optimize_meal(targets=high_protein_targets)
