#!/bin/bash
for job_id in {4864993..4865011}
do
	scancel $job_id
done