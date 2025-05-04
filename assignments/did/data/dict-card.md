# Difference in Differences (Card & Krueger, 1994)
This file is a dictionary of the data used in the paper *"Minimum Wages and Employment:
A Case Study of the Fast-Food Industry in New Jersey and Pennsylvania"* (Card & Krueger,
1994). This article is the most famous application of a Difference-in-Differences
design.

## Context
In April 1992, the state of New Jersey (NJ) raised the minimum wage from $4.25/hr to
$5.05/hr, while the neighboring state of Pennsylvania (PA) maintained the minimum wage
at $4.25/hr. Classic economic theory suggests that increasing the minimum wage will
reduce employment, and this setting is a natural experiment to empirically test
that hypothesis.

The authors surveyed fast-food restaurants in PA and NJ before and after the policy.
Using a DiD design, they compared the average number of employees between both groups
and found that the salary increase did not reduce employment in NJ. This seemingly
counterintuitive result is often referred to as the Card-Krueger paradox.

## Columns
The file `../data/card-krueger.csv` contains the raw data of the original paper (with
some minor changes, such as replacing `'.'` with `null`, etc.). The following lists
describe the meaning of each column:

### General columns
- `sheet`: sheet number (unique store id)
- `chain`: chain 1=bk; 2=kfc; 3=roys; 4=wendys
- `co_owned`: 1 if company owned
- `state`: 1 if NJ; 0 if Pa
- `southj`: 1 if in southern NJ
- `centralj`: 1 if in central NJ
- `northj`: 1 if in northern NJ
- `pa1`: 1 if in PA, northeast suburbs of Philadelphia
- `pa2`: 1 if in PA, Easton, etc
- `shore`: 1 if on NJ shore

### First interview
- `ncalls`: number of call-backs (0 if contacted on first call)
- `empft`: number of full-time employees
- `emppt`: number of part-time employees
- `nmgrs`: number of managers/assistant managers
- `wage_st`: starting wage ($/hr)
- `inctime`: months to usual first raise
- `firstinc`: usual amount of first raise ($/hr)
- `bonus`: 1 if cash bounty for new workers
- `pctaff`: % employees affected by new minimum wage
- `meals`: free/reduced price meals (0=none, 1=free, 2=reduced, 3=both)
- `open`: hour of opening
- `hrsopen`: number of hours open per day
- `psoda`: price of medium soda, including tax
- `pfry`: price of small fries, including tax
- `pentree`: price of entree, including tax
- `nregs`: number of cash registers in store
- `nregs11`: number of registers open at 11:00 am

### Second interview
- `type2`: type of second interview (1=phone; 2=personal)
- `status2`: status of second interview (0=refused, 1=answered, 2=renovation, 3=closed,
4=construction, 5=fire)
- `date2`: date of second interview (MMDDYY format)
- `ncalls2`: number of call-backs (2nd interview)
- `empft2`: number of full-time employees (2nd interview)
- `emppt2`: number of part-time employees (2nd interview)
- `nmgrs2`: number of managers/assistant managers (2nd interview)
- `wage_st2`: starting wage ($/hr) (2nd interview)
- `inctime2`: months to usual first raise (2nd interview)
- `firstin2`: usual amount of first raise ($/hr) (2nd interview)
- `special2`: 1 if special program for new workers
- `meals2`: free/reduced price meals (same codes as above)
- `open2r`: hour of opening (2nd interview)
- `hrsopen2`: number of hours open per day (2nd interview)
- `psoda2`: price of medium soda, including tax (2nd interview)
- `pfry2`: price of small fries, including tax (2nd interview)
- `pentree2`: price of entree, including tax (2nd interview)
- `nregs2`: number of cash registers (2nd interview)
- `nregs112`: number of registers open at 11:00 am (2nd interview)
