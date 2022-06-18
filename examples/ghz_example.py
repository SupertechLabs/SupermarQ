#!/usr/bin/env python
import time

import cirq_superstaq

import supermarq as sm


def main():
    service = cirq_superstaq.Service(
        api_key="""Insert superstaq token that you received from https://superstaq.super.tech""",
    )

    nq = 3
    ghz = sm.benchmarks.ghz.GHZ(nq)

    print(ghz.circuit())

    ibm_job = service.create_job(
        circuit=ghz.circuit(), repetitions=100, target="ibmq_qasm_simulator"
    )
    print("Created IBM job:", ibm_job)
    print("Current IBM status:", ibm_job.status())

    while ibm_job.status() != "Done":
        time.sleep(10)

    print("IBM Job status:", ibm_job.status())
    print("IBM Counts:", ibm_job.counts())
    print("IBM Benchmark score:", ghz.score(ibm_job.counts()))
    print("IBM Job_id:", ibm_job.job_id())
    ibm_job = service.get_job(ibm_job.job_id())
    print("IBM Counts:", ibm_job.counts())

    aws_job = service.create_job(circuit=ghz.circuit(), repetitions=100, target="aws_sv1_simulator")
    print("Created AWS job:", aws_job)
    print("Current AWS status:", aws_job.status())

    time.sleep(10)
    while aws_job.status() != "Done":
        time.sleep(10)

    print("AWS Job status:", aws_job.status())
    print("AWS Counts:", aws_job.counts())
    print("AWS Benchmark score:", ghz.score(aws_job.counts()))
    print("AWS Job_id:", aws_job.job_id())
    print("AWS Service.get_job:", service.get_job(aws_job.job_id()))


if __name__ == "__main__":
    main()
