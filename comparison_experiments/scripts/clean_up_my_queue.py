import os

uname = "danthes"
queue = "klab-gpu"

# get all jobs with DependencyNeverSatisfied
cmd = ["squeue", "-u", uname, "-p", queue]
res = os.popen(" ".join(cmd)).read().split("\n")[:-1]  # remove last empty line

print(res)

print("...")
print(res[-1])
print("...")


# get ids of jobs with DependencyNeverSatisfied
def get_never_satisfied():
    lines = [
        s for s in [r.split(" ") for r in res] if s[-1] == "(DependencyNeverSatisfied)"
    ]
    # drop all empty strings from each line
    lines = [[s for s in l if s != ""] for l in lines]
    # get job ids
    ids = [l[0] for l in lines]
    return ids


# cancel all jobs with DependencyNeverSatisfied

while len(get_never_satisfied()) > 0:
    ids = get_never_satisfied()
    for i in ids:
        cmd = ["scancel", i]
        res = os.popen(" ".join(cmd)).read()
        print(res)
