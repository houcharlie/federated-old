#!/bin/bash
for job_id in {4102451..4102550}
do
	scancel $job_id
done