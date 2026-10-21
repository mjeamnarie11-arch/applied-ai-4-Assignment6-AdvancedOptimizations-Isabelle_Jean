#!/usr/bin/env python3
"""
Warehouse optimization agent combining demand-based staffing,
frequency-based layout optimization, and greedy worker-to-order matching.

Staffing balances pending demand, worker utilization, and profit.
Layout swaps favor frequently retrieved products near delivery bays.
Order assignment balances pickup proximity and order value.

This agent uses optimization heuristics, not reinforcement learning.
 
"""

import numpy as np
from typing import Dict, Optional
from .standardized_agents import BaselineAgent
from environment.employee import EmployeeState
from environment.warehouse_grid import CellType

class SkeletonOptimizationAgent(BaselineAgent):
    """
    Coordinate staffing, product placement, and order pickup.
    Use demand, utilization, and distance estimates to support
    profitable operation while protecting managers and busy workers.
    """
    
    def __init__(self, env):
        super().__init__(env)
        self.name = "StudentOptimization"
        
        # Enable unlimited hiring for true economic optimization
        self.env._unlimited_hiring = True
        
        # Students can add their own state tracking here
        self.performance_metrics = []
        self.decision_history = []
        
        # Students can add algorithm parameters here
        # Example: Hungarian algorithm matrices, greedy search parameters, etc.
        self.staffing_parameters = {
            'hire_threshold_ratio': 3.0,
            'fire_threshold_ratio': 2.0,
            'profit_threshold': 0
        }
        
        # Students can implement adaptive parameters that change based on performance
        self.adaptive_optimization_enabled = False
        
    def reset(self):
        """Reset agent state - students should expand this"""
        self.action_history = []
        self.reward_history = []
        # TODO: Reset any neural network states, replay buffers, etc.
    
    def get_action(self, observation: Dict) -> Dict:
        """Coordinate staffing, layout, and worker assignments."""
        
        current_timestep = observation['time'][0]
        financial_state = observation['financial']
        queue_info = observation['order_queue']
        employee_info = observation['employees']

        staffing_action = self._get_naive_staffing_action(
            financial_state, employee_info
        )

        # Firing changes employee indices before assignments execute.
        # Allow automatic assignment for this step to avoid stale indices.
        if staffing_action == 2:
            assignments = [0] * 20
        else:
            assignments = self._get_naive_order_assignments(
                queue_info, employee_info
            )
        
        action = {
            'staffing_action': staffing_action,
            'layout_swap': self._get_naive_layout_action(current_timestep),
            'order_assignments': assignments
        }
        
        # TODO: Students should implement proper action recording for optimization
        self.action_history.append(action.copy())
        
        return action
    
    def _get_naive_staffing_action(self, financial_state, employee_info) -> int:
        """
        Adjust staffing using demand, utilization, and profit.
        Preserve at least two workers and protect the manager.
        Review staffing every 5 steps to respond to changing demand.
        """
        
        # TODO WEEK 2 STEP 1: Students should implement intelligent staffing logic
        # Current approach: Make random decisions based on "vibes"
        
        employees = self.env.employees
        workers = [employee for employee in employees
                   if not employee.is_manager]
        idle_workers = [worker for worker in workers
                        if worker.state == EmployeeState.IDLE]

        has_manager = any(employee.is_manager for employee in employees)
        can_hire = len(employees) < self.env.max_employees
        current_profit = float(financial_state[0])

        # Count orders that still need a worker.
        pending_orders = [
            order for order in self.env.order_queue.orders
            if order.status == "pending"
        ]
        queue_length = len(pending_orders)
        worker_count = len(workers)

        # The simulator fires the FIRST idle employee.
        # Only request firing when that person is a safe candidate.
        fire_candidate = next(
            (employee for employee in employees
             if employee.state == EmployeeState.IDLE),
            None
        )
        safe_to_fire = (
            fire_candidate is not None
            and not fire_candidate.is_manager
            and fire_candidate.current_order_id is None
            and worker_count > 2
        )

        # Maintain a minimum workforce so the warehouse can earn revenue.
        if worker_count < 2 and can_hire:
            return 1

        #  Review staffing every 5 steps to respond to changing demand.
        if len(self.action_history) % 5 != 0:
            return 0

        # Hire a manager without waiting for positive profit.
        if not has_manager:
            if can_hire:
                return 3
            if safe_to_fire:
                return 2  # Make room; hire the manager on a later review.

        queue_pressure = queue_length / max(1, worker_count)
        utilization = (
            (worker_count - len(idle_workers)) / max(1, worker_count)
        )

        # Hire when demand is high, workers are busy, and profit is positive.
        if (
            can_hire
            and queue_pressure > 3.0
            and utilization >= 0.80
            and current_profit > 0
        ):
            return 1

        # Reduce wage costs only when there is spare capacity.
        severe_overstaffing = (
            queue_length == 0
            and len(idle_workers) > worker_count / 2
        )
        if (
            safe_to_fire
            and queue_pressure < 1.5
            and utilization < 0.50
            and (current_profit < 0 or severe_overstaffing)
        ):
            return 2

        return 0
    
    def _get_naive_layout_action(self, current_timestep) -> list:
        """
        Every 100 steps, select a beneficial accessible storage swap.
        Score retrieval frequency per product copy, delivery distance,
        and manager travel. Require an idle manager.
        """
        
        # TODO WEEK 1 STEP 1: Students should implement intelligent layout optimization
        # Current approach: Occasionally make random swaps
        
        if current_timestep == 0 or current_timestep % 100 != 0:
            return [0, 0]

        grid = self.env.warehouse_grid
        frequencies = grid.item_access_frequency

        completed_moves = sum(
            relocation["status"] == "completed"
            for relocation in self.env.relocations_history
        )

        print(
            f"[Layout step {current_timestep}] "
            f"Recorded accesses: {frequencies.sum():.0f}, "
            f"Employees: {len(self.env.employees)}, "
            f"Relocations started: {len(self.env.relocations_history)}, "
            f"Completed: {completed_moves}"
        )

        # The environment uses the first manager for relocation requests.
        manager = next(
            (employee for employee in self.env.employees
             if employee.is_manager),
            None
        )

        if manager is None:
            print("  Skipped: no manager available.")
            return [0, 0]

        if manager.state != EmployeeState.IDLE:
            print(f"  Skipped: manager is busy ({manager.state.name}).")
            return [0, 0]

        if not np.any(frequencies > 0):
            print("  Skipped: no product retrievals recorded yet.")
            return [0, 0]

        # Only accessible permanent storage cells are valid swap locations.
        positions = [
            (x, y)
            for y in range(grid.height)
            for x in range(grid.width)
            if grid.cell_types[y, x] == CellType.STORAGE.value
            and grid.can_access_storage(x, y)
        ]

        distances = {
            position: self._distance_to_delivery(position)
            for position in positions
        }

        # Estimate demand per copy of each product.
        copy_counts = np.bincount(
            grid.item_grid[grid.item_grid >= 0],
            minlength=len(frequencies)
        )

        best_score = 0.0
        best_swap = [0, 0]

        for source in positions:
            source_item = grid.get_item_at_position(*source)
            if source_item is None:
                continue

            source_frequency = (
                frequencies[source_item] / max(1, copy_counts[source_item])
            )
            if source_frequency <= 0:
                continue

            for target in positions:
                distance_saved = distances[source] - distances[target]

                # Require at least two steps of delivery-distance improvement.
                if distance_saved < 2:
                    continue

                target_item = grid.get_item_at_position(*target)
                if target_item == source_item:
                    continue

                target_frequency = 0.0
                if target_item is not None:
                    target_frequency = (
                        frequencies[target_item]
                        / max(1, copy_counts[target_item])
                    )

                # Account for moving the other product farther from delivery.
                net_benefit = (
                    source_frequency - target_frequency
                ) * distance_saved

                if net_benefit <= 0:
                    continue

                # Favor beneficial swaps with shorter manager travel.
                relocation_distance = (
                    grid.manhattan_distance(manager.position, source)
                    + grid.manhattan_distance(source, target)
                )
                score = net_benefit / (1 + relocation_distance)

                if score > best_score:
                    best_score = score
                    best_swap = [
                        source[1] * grid.width + source[0],
                        target[1] * grid.width + target[0]
                    ]

        print(
            f"  Selected swap: {best_swap}, "
            f"Estimated score: {best_score:.2f}"
        )

        if best_swap == [0, 0]:
            print("  No beneficial swap found.")

        return best_swap

    def _distance_to_delivery(self, position) -> int:
        """Return Manhattan distance to the nearest truck bay."""
        grid = self.env.warehouse_grid
        return min(
            grid.manhattan_distance(position, bay)
            for bay in grid.truck_bay_positions
        )
    
    def _get_naive_order_assignments(self, queue_info, employee_info) -> list:
        """
        Greedily match idle workers to pending orders

        Score proximity to the nearest required item and order value
        Each worker and order can be selected only once per step
        """
        
        # TODO WEEK 2 STEP 2: Students should implement intelligent order assignment
        # Match idle workers using pickup proximity and order value
        
        assignments = [0] * 20  # No assignments by default
        
        grid = self.env.warehouse_grid

        idle_workers = [
            (index, employee)
            for index, employee in enumerate(self.env.employees)
            if (
                not employee.is_manager
                and employee.state == EmployeeState.IDLE
                and employee.current_order_id is None
            )
        ]

        # Preserve queue indices: each action slot refers to that position.
        pending_orders = [
            (index, order)
            for index, order in enumerate(
                self.env.order_queue.orders[:len(assignments)]
            )
            if order.status == "pending"
        ]

        if not idle_workers or not pending_orders:
            return assignments

        max_value = max(
            1.0,
            max(order.value for _, order in pending_orders)
        )

        # Look up each product once during this decision.
        required_items = {
            item
            for _, order in pending_orders
            for item in order.items
        }
        item_locations = {
            item: [
                position
                for position in grid.find_item_locations(item)
                if grid.can_access_storage(*position)
            ]
            for item in required_items
        }

        scored_pairs = []

        for order_index, order in pending_orders:
            # A relocation can temporarily make a product unavailable.
            if not order.items or any(
                not item_locations[item] for item in order.items
            ):
                continue

            for worker_index, worker in idle_workers:
                distance = min(
                    abs(worker.position[0] - position[0])
                    + abs(worker.position[1] - position[1])
                    for item in order.items
                    for position in item_locations[item]
                )

                distance_score = 1.0 / (1.0 + distance)
                value_score = order.value / max_value
                score = 0.70 * distance_score + 0.30 * value_score

                scored_pairs.append(
                    (score, order_index, worker_index)
                )

        # Prefer higher scores; use indices to break ties consistently.
        scored_pairs.sort(key=lambda pair: (-pair[0], pair[1], pair[2]))

        assigned_workers = set()
        assigned_orders = set()

        for score, order_index, worker_index in scored_pairs:
            if (
                worker_index in assigned_workers
                or order_index in assigned_orders
            ):
                continue

            # The simulator uses 1-based employee indices; zero means none.
            assignments[order_index] = worker_index + 1
            assigned_workers.add(worker_index)
            assigned_orders.add(order_index)

        return assignments
    
    def record_reward(self, reward: float):
        """
        WEEK 3 STEP 1: Reward tracking and learning - students should expand this
        
        TODO WEEK 3 STEP 1: Students should implement:
        - Proper reward tracking
        - Performance analysis
        - Adaptive parameter adjustment
        - Multi-objective optimization
        """
        self.reward_history.append(reward)
        
        # Skeleton optimization - doesn't actually learn anything useful
        if len(self.reward_history) > 10:
            # "Update" weights randomly (this doesn't actually improve performance)
            if reward > 0:
                self.staffing_weights += np.random.randn(4) * 0.01
                self.layout_weights += np.random.randn(3) * 0.01
    
    def should_update_policy(self) -> bool:
        """
        WEEK 3 STEP 2: Policy updates and adaptation - students should improve this
        
        TODO WEEK 3 STEP 2: Students should implement proper update schedules
        """
        return len(self.action_history) % 50 == 0  # Arbitrary update frequency
    
    def get_performance_metrics(self) -> Dict:
        """
        Get agent performance metrics for analysis
        Students can use this to debug their improvements
        """
        if not self.reward_history:
            return {"avg_reward": 0, "total_actions": 0}
        
        return {
            "avg_reward": np.mean(self.reward_history[-100:]),  # Last 100 rewards
            "total_actions": len(self.action_history),
            "exploration_rate": self.exploration_rate,
            "recent_performance": np.mean(self.reward_history[-10:]) if len(self.reward_history) >= 10 else 0
        }


def create_skeleton_optimization_agent(env) -> SkeletonOptimizationAgent:
    """Factory function to create skeleton Optimization agent"""
    return SkeletonOptimizationAgent(env)


# TODO: Students should implement these advanced components:

class StudentOptimizationAgent(SkeletonOptimizationAgent):
    """
    Template for students to implement their improved Optimization agent
    
    Suggested improvements:
    1. Replace random staffing with demand-based hiring
    2. Implement proper layout optimization using item frequencies
    3. Add distance-based order assignment
    4. Implement basic Q-optimization or policy gradients
    5. Add proper exploration vs exploitation balance
    """
    
    def __init__(self, env):
        super().__init__(env)
        self.name = "StudentOptimization"
        
        # TODO: Students implement these
        # self.q_table = {}
        # self.policy_network = SimpleNeuralNetwork()
        # self.experience_buffer = []
        # self.target_network = None
        
    def _get_improved_staffing_action(self, financial_state, employee_info, queue_info):
        """
        WEEK 2 STEP 1: Students implement intelligent staffing:
        - Hire when queue is growing
        - Fire when queue is empty for extended periods
        - Consider profit margins before hiring
        - Balance managers vs workers
        """
        pass
    
    def _get_improved_layout_action(self, observation):
        """
        WEEK 1 STEP 1: Students implement smart layout optimization:
        - Move frequently accessed items closer to delivery
        - Group items that are often ordered together
        - Only optimize when queue is manageable
        - Track swap effectiveness
        """
        pass
    
    def _get_improved_order_assignments(self, queue_info, employee_info):
        """
        WEEK 2 STEP 2: Students implement efficient order assignment:
        - Assign orders to closest available employees
        - Prioritize high-value or urgent orders
        - Consider employee current locations
        - Balance workload across employees
        """
        pass
    
    def learn_from_experience(self, state, action, reward, next_state, done):
        """
        WEEK 3 STEP 3: Students implement multi-objective optimization:
        - Performance trend analysis
        - Adaptive parameter tuning
        - Multi-objective trade-off handling
        - Robust optimization techniques
        """
        pass