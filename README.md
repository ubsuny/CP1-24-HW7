# Roots and Integrals

This Homework is seperated into two task groups (algorithm, numerical calculation and presentation) and the same rules apply as in HW4.
Remember you are in charge to have a review assigned if one available, also you have to finish your review 2 hours after assignment.
Once the review is finished the original author has 12 hours to fix the issues mentionend in the review.

## Task group 1 algorithm (12 members per algorithm)
Use the four integral algorithms (simpson, adaptive trapezoid, trapezoid) to create a Jupyter notebook that imports the functions from `calculus.py` and use them to calculate and plot the integral of:

In common files `calculus.py` and `test_calculus.py` create functions and unit tests to calculate the integral using the following algorithm :
- simpson
- adaptive trapezoid
- trapezoid

Each algorithm has to be implemented in three ways.
1. wrapper for numpy implementation (due Wednesday)
2. pure python implementation (due Friday)
3. c/c++ implementation using ctypes in python (due Monday)

---

In the same file `calculus.py` and `test_calculus.py` implement three root-finding algorithms and unit tests:

- bisection
- secant
- tangent

Each algorithm has to be implemented in three ways.
1. wrapper for numpy implementation (due Wednesday)
2. pure python implementation (due Friday)
3. c/c++ implementation using ctypes in python (due Monday)

---

**Hint:** You can use https://github.com/ubsuny/CompPhys/blob/main/Calculus/ for examples for the C/C++ code

Generate one PR per implementation.
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
y(x) = \tanh(x).
$$

You can only use functions from `calculus.py`.
Compare the accuracies (how many digits are correct) and efficiencies (how many steps does it take to reach a given accuracy) for each algorithm and each implementation as it becomes available.
Generate a figure `function_name.png` showing the function and mark all roots with circles for each algorithm including the number of steps inside in the figure.

---

Generate one PR per function.
Review each others PR.

## Task group 4 maintainers (max 1 member)
- Reuse github actions for linting and unit tests
- Merge PR
- assign Reviews after member requests
  
---
## Grading

| Homework Points                  |                |              |            |
| -------------------------------- | -------------- | ------------ | ---------- |
|                                  |                |              |            |
| Interaction on project           |                |              |            |
| Category                         | min per person | point factor | max points |
| Commits                          | 1              | 1            | 1          |
| Pull requests                    | 3              | 2            | 6          |
| PR Accepted                      | 3              | 3            | 9          |
| Other PR reviewed (by request)   | 3              | 4            | 12          |     
| Issues                           | 0              | 0            | 0          | 
| Closed Issues                    | 0              | 0            | 0          |
| \# Conversations                 | 12             | 1/4          | 3          |
|                                  |                |              |            |
| Total                            |                |              | 31         |
|                                  |                |              |            |
| Shared project points            |                |              |            |
| \# Milestones                    | 12             | 1/4          | 3          |
|                                  |                |              |            |
| Total                            |                |              | 34         |
|                                  |                |              |            |
|                                  |                |              |            |
| Result                           |                |              |            |
| Task completion                  |                |              | 8          |
|                                  |                |              |            |
| Sum                              |                |              | 42         |
