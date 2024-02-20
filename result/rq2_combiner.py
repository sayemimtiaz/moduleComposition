import os
import csv

from util.common import load_combos

base_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

out = open(os.path.join(base_path, "result", "rq2" + ".csv"), "w")
out.write(
    'Combination ID,MS Train,MU Train,Scratch Train,SC Infer,Scratch Infer,Voting Infer,MU Setup,SC Setup\n')
muFile = os.path.join(base_path, "result", "mu" + ".csv")
staticFile = os.path.join(base_path, "result", "static" + ".csv")
msFile = os.path.join(base_path, "result", "ms" + ".csv")
scratchFile = os.path.join(base_path, "result", "scratch" + ".csv")
scFile = os.path.join(base_path, "result", "sc" + ".csv")

static_infer = {}
with open(staticFile) as csv_file:
    csv_reader = csv.reader(csv_file)
    next(csv_reader)

    for (serial, row) in enumerate(csv_reader):
        static_infer[row[0].strip()] = round(float(row[2].strip()) * 1000000.0, 2)

ms_time = {}
with open(msFile) as csv_file:
    csv_reader = csv.reader(csv_file)
    next(csv_reader)

    for (serial, row) in enumerate(csv_reader):
        ms_time[row[0].strip()] = round(float(row[2].strip()), 2)

mu_time = {}
mu_setup_time = {}
with open(muFile) as csv_file:
    csv_reader = csv.reader(csv_file)
    next(csv_reader)

    for (serial, row) in enumerate(csv_reader):
        mu_time[row[0].strip()] = round(float(row[3].strip()), 2)
        mu_setup_time[row[0].strip()] = round(float(row[2].strip()), 2)

scratch_infer = {}
scratch_time = {}
with open(scratchFile) as csv_file:
    csv_reader = csv.reader(csv_file)
    next(csv_reader)

    for (serial, row) in enumerate(csv_reader):
        scratch_infer[row[0].strip()] = round(float(row[3].strip()) * 1000000.0, 2)
        scratch_time[row[0].strip()] = round(float(row[2].strip()), 2)

sc_infer = {}
sc_setup={}
with open(scFile) as csv_file:
    csv_reader = csv.reader(csv_file)
    next(csv_reader)

    for (serial, row) in enumerate(csv_reader):
        sc_infer[row[0].strip()] = round(float(row[3].strip()) * 1000000.0, 2)
        sc_setup[row[0].strip()] = round(float(row[2].strip()), 2)

comboList = load_combos(start=0, end=199)

ds_sum = {}
mod_sum = {}

for cmbId in mu_time.keys():
    out.write(str(cmbId) + ',' + str(ms_time[cmbId]) + ',' + str(mu_time[cmbId]) + ','+str(scratch_time[cmbId])+','
              + str(sc_infer[cmbId]) + ',' + str(scratch_infer[cmbId]) + ','
              + str(static_infer[cmbId])
              + ',' + str(mu_setup_time[cmbId])
              + ',' + str(sc_setup[cmbId])
              + '\n')


out.close()

