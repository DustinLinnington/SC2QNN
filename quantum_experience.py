from IBMQuantumExperience import IBMQuantumExperience
import qiskit

qr = qiskit.QuantumRegister(1)
cr = qiskit.ClassicalRegister(1)
program = qiskit.QuantumCircuit(qr, cr)
program.measure(qr, cr)

qiskit.IBMQ.load_accounts()
backend = qiskit.backends.ibmq.least_busy(qiskit.IBMQ.backends(simulator=False))
print("Using the least busy device:", backend.name())
job = qiskit.execute(program, backend)
# job = qiskit.execute(program, qiskit.Aer.get_backend('qasm_simulator'))

print(job.result().get_counts())

# USING A DICTIONARY TO CREATE THE PROGRAM
# Q_SPECS = {
#     "name": "Program-tutorial",
#     "circuits": [{
#          "name": "Circuit",
#          "quantum_registers": [{
#              "name":"qr",
#              "size": 4 
#          }],
#          "classical_registers": [{
#               "name":"cr",
#               "size": 4
#          }]}],
# }
# Q_program = QuantumProgram(specs=Q_SPECS) 