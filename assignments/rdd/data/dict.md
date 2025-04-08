# Class Size and Academic Performance
The paper _Using **Maimonides**' Rule to Estimate the Effect of Class Size on
Scholastic Achievement_ (Angrist and Lavy, 1999) studies the effect of class
size on academic performance.

## Research Question
The authors use data from fifth grade Israeli schools to study the impact of
class size on academic performance. In Israel, school groups are split in two or
more groups if there's more than 40 students in a school grade (each resulting
group can have no more than 40 students).

Since there's nothing systematically different between a group of 40 students
and another of 41, the difference between test results can be attributed to the
difference in class size. The authors use an RDD to study the local treatment
effect of classroom size on academic performance.

### Data
There are 172 observations in total. Each one represents a school. Schools are
paired according to the share of students with dissabilities to further refine
the experiment. For each pair of comparable schools, one has slightly more than
40 students (control) and the other has 40 students or less (treatment).

Our goal is to prove that the treatment group has significantly lower academic
scores than the control group.

### Columns
- `scode`: School ID
- `numclass`: Number of classes in the fifth grade
- `cohsize`: Total number of students in the fifth grade, near 40 for these
schools.
- `avgmath`: Average grade in math in the fifth grade.
- `avgverb`: Average grade in verbal in the fifth grade.
- `tipuach`: Percent of disadvantaged students (used to form matched pairs).
- `clasz`: Average class size in the fifth grade (equal to 
`cohsize / numclass`).
- `z`: 1 if `cohsize <= 40`, 0 if `cohsize > 40`.
- `pair`: pair ID, 1, 2, ..., 86

## Source
The [original data](
    https://rdrr.io/cran/DOS/man/angristlavy.html
) is part of the [DOS R package](
    https://cran.r-project.org/package=DOS
).
