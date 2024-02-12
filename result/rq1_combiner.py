import os
import csv

from util.common import load_combos

base_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

out = open(os.path.join(base_path, "result", "rq1" + ".csv"), "w")
out.write(
    'Combination ID,Scratch,Static,SC,MU,MS,'
    '#Dataset,#Modules\n')
muFile = os.path.join(base_path, "result", "mu" + ".csv")
staticFile = os.path.join(base_path, "result", "static" + ".csv")
msFile = os.path.join(base_path, "result", "ms" + ".csv")
scratchFile = os.path.join(base_path, "result", "scratch" + ".csv")
scFile = os.path.join(base_path, "result", "sc" + ".csv")

static_acc = {}
with open(staticFile) as csv_file:
    csv_reader = csv.reader(csv_file)
    next(csv_reader)

    for (serial, row) in enumerate(csv_reader):
        static_acc[row[0].strip()] = round(float(row[1].strip()) * 100.0, 2)

ms_acc = {}
with open(msFile) as csv_file:
    csv_reader = csv.reader(csv_file)
    next(csv_reader)

    for (serial, row) in enumerate(csv_reader):
        ms_acc[row[0].strip()] = round(float(row[1].strip()) * 100.0, 2)

mu_acc = {}
with open(muFile) as csv_file:
    csv_reader = csv.reader(csv_file)
    next(csv_reader)

    for (serial, row) in enumerate(csv_reader):
        mu_acc[row[0].strip()] = round(float(row[1].strip()) * 100.0, 2)

scratch_acc = {}
with open(scratchFile) as csv_file:
    csv_reader = csv.reader(csv_file)
    next(csv_reader)

    for (serial, row) in enumerate(csv_reader):
        scratch_acc[row[0].strip()] = round(float(row[1].strip()) * 100.0, 2)

sc_acc = {}
with open(scFile) as csv_file:
    csv_reader = csv.reader(csv_file)
    next(csv_reader)

    for (serial, row) in enumerate(csv_reader):
        sc_acc[row[0].strip()] = round(float(row[1].strip()) * 100.0, 2)

comboList = load_combos(start=0, end=199)

ds_sum = {}
mod_sum = {}

for cmbId in mu_acc.keys():

    no_ds = set()
    no_mod = len(comboList[int(cmbId)])
    for (_d, _c, _m) in comboList[int(cmbId)]:
        no_ds.add(_d)
    no_ds = len(list(no_ds))

    static = static_acc[cmbId]
    ms = ms_acc[cmbId]
    scratch = scratch_acc[cmbId]
    mu = mu_acc[cmbId]
    sc = sc_acc[cmbId]

    out.write(str(cmbId) + ',' + str(scratch) + ',' + str(static) + ','
              + str(sc) + ',' + str(mu) + ','
              + str(ms)
              + ',' + str(no_ds)
              + ',' + str(no_mod)
              + '\n')

    if no_ds not in ds_sum:
        ds_sum[no_ds] = (0, 0, 0, 0, 0, 0)
    if no_mod not in mod_sum:
        mod_sum[no_mod] = (0, 0, 0, 0, 0, 0)

    ds_sum[no_ds] = (ds_sum[no_ds][0] + 1,
                     ds_sum[no_ds][1] + scratch,
                     ds_sum[no_ds][2] + static,
                     ds_sum[no_ds][3] + sc,
                     ds_sum[no_ds][4] + mu,
                     ds_sum[no_ds][5] + ms
                     )

    mod_sum[no_mod] = (mod_sum[no_mod][0] + 1,
                       mod_sum[no_mod][1] + scratch,
                       mod_sum[no_mod][2] + static,
                       mod_sum[no_mod][3] + sc,
                       mod_sum[no_mod][4] + mu,
                       mod_sum[no_mod][5] + ms
                       )

out.close()

out = open(os.path.join(base_path, "result", "ds_stat.csv"), "w")
out.write('#Dataset,Scratch,Static,SC,MU,MS\n')
ds_sum = dict(sorted(ds_sum.items()))
for no_ds in ds_sum:
    scratch = ds_sum[no_ds][1] / ds_sum[no_ds][0]
    static = ds_sum[no_ds][2] / ds_sum[no_ds][0]
    sc = ds_sum[no_ds][3] / ds_sum[no_ds][0]
    mu = ds_sum[no_ds][4] / ds_sum[no_ds][0]
    ms = ds_sum[no_ds][5] / ds_sum[no_ds][0]

    out.write(str(no_ds) + ',' + str(scratch) + ',' + str(static) + ','
              + str(sc) + ',' + str(mu)+',' + str(ms) + '\n')
out.close()

mod_sum = dict(sorted(mod_sum.items()))

out = open(os.path.join(base_path, "result", "mod_stat.csv"), "w")
out.write('#Module,Scratch,Static,SC,MU,MS\n')

for no_ds in mod_sum:
    scratch = mod_sum[no_ds][1] / mod_sum[no_ds][0]
    static = mod_sum[no_ds][2] / mod_sum[no_ds][0]
    sc = mod_sum[no_ds][3] / mod_sum[no_ds][0]
    mu = mod_sum[no_ds][4] / mod_sum[no_ds][0]
    ms = mod_sum[no_ds][5] / mod_sum[no_ds][0]

    out.write(str(no_ds) + ',' + str(scratch) + ',' + str(static) + ','
              + str(sc) + ',' + str(mu)+',' + str(ms) + '\n')
out.close()
