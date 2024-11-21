# Roots and Integrals

This Homework is seperated into two task groups (algorithm, numerical calculation and presentation) and the same rules apply as in HW4.
Remember you are in charge to have a review assigned if one available, also you have to finish your review 2 hours after assignment.
Once the review is finished the original author has 12 hours to fix the issues mentionend in the review.

## Task group 1 algorithm (2 members per algorithm)
Use the four integral algorithms (simpson, adaptive trapezoid, trapezoid) to calculate and plot the integral of:

In common files `calculus.py` and `test_calculus.py` create functions and unit tests to calculate the integral using the following algorithm :
- simpson
- adaptive trapezoid
- trapezoid

Each algorithm has to be implemented in three ways.
1. wrapper for numpy (only trapezoids) and scipy (only simpson) implementation (due Wednesday)
2. pure python implementation (due Friday)
3. c/c++ implementation using ctypes in python (due Monday)

---

In the same file `calculus.py` and `test_calculus.py` implement three root-finding algorithms and unit tests:

- bisection
- secant
- tangent

Each algorithm has to be implemented in three ways.
1. wrapper for scipy implementation (due Wednesday)
2. pure python implementation (due Friday)
3. c/c++ implementation using ctypes in python (due Monday)

---

**Hint:** You can use https://github.com/ubsuny/CompPhys/blob/main/Calculus/ for examples for the C/C++ code

Generate one PR per implementation and one PR for unit testing.
Review each others PR.

## Task group 2 numerical calculation and presentation (max 4 members due till Monday)

Calculate the following integrals (max 2 members):

- $\exp(-1/x)$
- $\cos(1/x)$
- $x^3 + 1$

with the according boundaries: 

$$
[0,10],(0,3\pi],[-1,1]
$$

You can only use functions from `calculus.py`.
Compare the accuracies (how many digits are correct) and efficiencies (how many steps does it take to reach a given accuracy) for each algorithm and each implementation as it becomes available.
Generate a figure `function_name.png` showing the function and color the area of integral to be calculated with its value for each algorithm including the number of steps inside in the figure.

---

Compare the accuracies (how many digits are correct) and efficiencies (how many steps does it take to reach a given accuracy) for all root-finding algorithms and implementations on the functions (max 2 members):

$$
y(x) = \frac 1 {\sin(x)} \textrm{ and}
$$

$$
y(x) = \tanh(x) \textrm{ and}
$$

$$
y(x) = \sin(x).
$$

You can only use functions from `calculus.py`.
Compare the accuracies (how many digits are correct) and efficiencies (how many steps does it take to reach a given accuracy) for each algorithm and each implementation as it becomes available.
Generate a figure `function_name.png` showing the function and mark all roots with circles for each algorithm including the number of steps inside in the figure.

---

Generate one PR per function and one PR for unit testing.
Review each others PR.

## Task group 3 maintainers (max 1 member)
- Maintainer is in charge of the directory structure
- Merge and review PRs for complience
  
---
## Grading

| Homework Points                  |                |              |            |
| -------------------------------- | -------------- | ------------ | ---------- |
|                                  |                |              |            |
| Interaction on project           |                |              |            |
| Category                         | min per person | point factor | max points |
| Commits                          | 1              | 1            | 1          |
| Pull requests                    | 2              | 2            | 4          |
| PR Accepted                      | 2              | 4            | 8          |
| Other PR reviewed.               | 2              | 3            | 6          |     
| Issues                           | 0              | 0            | 0          | 
| Closed Issues                    | 0              | 0            | 0          |
| \# Conversations                 | 12             | 1/4          | 3          |
|                                  |                |              |            |
| Total                            |                |              | 20         |
|                                  |                |              |            |
| Shared project points            |                |              |            |
| \# Milestones                    | 12             | 1/4          | 3          |
|                                  |                |              |            |
| Total                            |                |              | 23         |
|                                  |                |              |            |
|                                  |                |              |            |
| Result                           |                |              |            |
| Task completion                  |                |              | 19         |
|                                  |                |              |            |
| Sum                              |                |              | 42         |
