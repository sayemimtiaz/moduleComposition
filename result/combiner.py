import os
import csv

model_suffix = '1'
base_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

out = open(os.path.join(base_path, "result", "combined_model" + model_suffix + ".csv"), "w")
out.write(
    'Serial,DeepCompose Accuracy,Static Composition Accuracy,Module Stacking Accuracy,Train-from-scratch Accuracy,'
    'DeepCompose Training Time,'
    'Module Stacking Training Time,Train-from-scratch Training Time,'
    '#Dataset,#Modules,#MNIST,#EMNIST,#FMNIST,#KMNIST,Combination\n')
updateFile = os.path.join(base_path, "result", "update_model" + model_suffix + ".csv")
staticFile = os.path.join(base_path, "result", "static_model" + model_suffix + ".csv")
stackingFile = os.path.join(base_path, "result", "dynamic_model" + model_suffix + ".csv")

static_acc = {}
with open(staticFile) as csv_file:
    csv_reader = csv.reader(csv_file)
    next(csv_reader)

    for (serial, row) in enumerate(csv_reader):
        static_acc[serial] = round(float(row[1].strip()) * 100.0, 2)

stacking_acc = {}
stacking_time = {}
with open(stackingFile) as csv_file:
    csv_reader = csv.reader(csv_file)
    next(csv_reader)

    for (serial, row) in enumerate(csv_reader):
        stacking_acc[serial] = round(float(row[1].strip()) * 100.0, 2)
        stacking_time[serial] = round(float(row[2].strip()) * 100.0, 2)

ds_sum = {}
mod_sum = {}
with open(updateFile) as csv_file:
    csv_reader = csv.reader(csv_file)
    next(csv_reader)

    for (serial, row) in enumerate(csv_reader):
        tcmb = row[0].strip()

        tcmb = tcmb.replace('(', '')
        tcmb = tcmb.split(')')

        no_ds = len(tcmb) - 1
        no_mod = 0
        mod_count = {'mnist': 0, 'fmnist': 0, 'emnist': 0, 'kmnist': 0}
        for _c in tcmb[:-1]:
            tc = _c.split(':')
            tcc = tc[0]
            tc = tc[1].split('-')[:-1]
            no_mod += len(tc)
            mod_count[tcc.strip()] = len(tc)

        static = static_acc[serial]
        stacking = stacking_acc[serial]
        update = round(float(row[1].strip()) * 100.0, 2)
        scratch = round(float(row[2].strip()) * 100.0, 2)
        mod_time = round(float(row[4].strip()) * 100.0, 2)
        scratch_time = round(float(row[5].strip()) * 100.0, 2)

        out.write(str(serial) + ',' + str(update) + ',' + str(static) + ','
                  + str(stacking) + ',' + str(scratch) + ','
                  + str(mod_time) + ',' + str(stacking_time[serial])
                  + ',' + str(scratch_time)
                  + ',' + str(no_ds)
                  + ',' + str(no_mod)
                  + ',' + str(mod_count['mnist'])
                  + ',' + str(mod_count['emnist'])
                  + ',' + str(mod_count['fmnist'])
                  + ',' + str(mod_count['kmnist'])
                  + ',' + str(row[0])
                  + '\n')

        if no_ds not in ds_sum:
            ds_sum[no_ds] = (0, 0, 0, 0, 0)
        if no_mod not in mod_sum:
            mod_sum[no_mod] = (0, 0, 0, 0, 0)

        ds_sum[no_ds] = (ds_sum[no_ds][0] + 1,
                         ds_sum[no_ds][1] + update,
                         ds_sum[no_ds][2] + static,
                         ds_sum[no_ds][3] + stacking,
                         ds_sum[no_ds][4] + scratch
                         )

        mod_sum[no_mod] = (mod_sum[no_mod][0] + 1,
                           mod_sum[no_mod][1] + update,
                           mod_sum[no_mod][2] + static,
                           mod_sum[no_mod][3] + stacking,
                           mod_sum[no_mod][4] + scratch)

out.close()

out = open(os.path.join(base_path, "result", "ds_stat.csv"), "w")
out.write('#Dataset,Static,Update,Stacking,Scratch\n')

ds_sum = dict(sorted(ds_sum.items()))
for no_ds in ds_sum:
    update = ds_sum[no_ds][1] / ds_sum[no_ds][0]
    static = ds_sum[no_ds][2] / ds_sum[no_ds][0]
    stacking = ds_sum[no_ds][3] / ds_sum[no_ds][0]
    scratch = ds_sum[no_ds][4] / ds_sum[no_ds][0]

    out.write(str(no_ds) + ',' + str(static) + ',' + str(update) + ','
              + str(stacking) + ',' + str(scratch) + '\n')
out.close()

mod_sum = dict(sorted(mod_sum.items()))

out = open(os.path.join(base_path, "result", "mod_stat.csv"), "w")
out.write('#Module,Static,Update,Stacking,Scratch\n')
for no_ds in mod_sum:
    update = mod_sum[no_ds][1] / mod_sum[no_ds][0]
    static = mod_sum[no_ds][2] / mod_sum[no_ds][0]
    stacking = mod_sum[no_ds][3] / mod_sum[no_ds][0]
    scratch = mod_sum[no_ds][4] / mod_sum[no_ds][0]

    out.write(str(no_ds) + ',' + str(static) + ',' + str(update) + ','
              + str(stacking) + ',' + str(scratch) + '\n')
out.close()
