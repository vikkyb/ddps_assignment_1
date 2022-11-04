import numpy as np
import math
import copy

np.random.seed(1234)

"""
TODO list
1. Data locality + punishment (DONE)
2. Add kill functionality and kill percentage (DONE)
3. Log waiting and processing times (DONE)
4. Fair scheduling (DONE)
5. Delay fair scheduling (DONE)
6. Redistribution FIFO after task finishing frees up partial space (uncertain whether to implement)
7. Document code
"""

### CONSTANTS TO BE SET ###
# These constants are important for how fast jobs are processed and change how the queue builds up over time
# If you want to change queue behaviour and prevent excessive queue buildup, change these values 
n_nodes = 60
experiment_duration = 100000 # Can be exceeded if run_until_finished
job_frequency = 1/200

# These values are important to selecting what system you want to simulate
run_until_finished = False
scheduler_type = "delay_fair" # fifo, fair, delay_fair

# If you use delay scheduling, set this parameter for the max_skip count before a job is forced to join
skip_count_max = 100

# These constants indicate how data will be distrbuted over the nodes
# Data of a task will be present on data_copies / data_splits * n_nodes nodes
n_data_splits = 6
n_data_copies = 3

# Set this to true to print frequent updates during simulation
show_progress = False

### CONSTANTS TO BE SET ###


### Create job types ###
# {name : [occurrence weight, mean duration, stdev duration, transfer punishment]}
job_types = {
    "calculate_pi_decimals":    [30,     1000,   200,    10],
    "newton_rhapton_solver":    [20,     1500,   200,    20],
    "linear_regression":        [15,     2200,   500,    50],
    "mnist_classifier":         [10,     5000,   2000,   100],
    "train_imagenet":           [3,      13000,  3000,   2000],
    "train_gpt_6":              [1,      25000,  7000,   3000],
    "answer_to_the_universe":   [1,      60000,  10000,  10000]}
### Create job types ###

class Forecaster:
    """
    This class creates the job iternary, deciding beforehand which
    job will be triggered at what step. 
    """
    def __init__(self, scheduler):
        """
        Initialize forecaster

        input: A Scheduler class object
        """

        self.scheduler = scheduler
        self.future_jobs = {}

        self.normalize_probabilities()
        
    def forecast_future_jobs(self):
        """
        Create a dictionary where each key is an integer 'step'  
        (indicating when a job is queued) and where each value
        is a list of string(s) that indicate the job type.

        return: future_jobs (list)
        """

        for i in range(experiment_duration):
            if np.random.random() <= job_frequency:
                self.future_jobs[i] = self.sample_job()
        # print("{0} jobs scheduled".format(len(self.future_jobs)))
        return self.future_jobs
    
    def normalize_probabilities(self):
        """
        Normalizes the probabilities received from global job_types
        """
        norm = 0
        for i in job_types:
            norm += job_types[i][0]
        self.job_probs = {}
        for i in job_types:
            self.job_probs[i] = job_types[i][0] / norm

    def sample_job(self):
        """
        Samples a job randomly based on normalized probabilities
        
        return: job_name (str)
        """
        rgn = np.random.random()
        c = 0
        for job, prob in self.job_probs.items():
            c += prob
            if rgn <= c:
                return job        

class Task:
    """
    A Task is a small part of a job, being performed on a node.
    The goal of the task is to count down the assigned ticks to 0 and when finished,
    it will subtract its performed work from the parent job. 
    Punishment ticks are accounted for when data needs to be transferred, but the punishment will
    not be subtracted from the parent job.
    """
    def __init__(self, job, assigned_ticks, node):
        """
        Initalize the Task and calculate the total ticks to process and bind the task to its parent job as well as the supplied node.
        This function somewhat overlaps with the node.assign() function and can therefore be optimized/rewritten at some point.
        
        Inputs: job (Job class), assigned_ticks (int), node (Node class)
        """
        self.job = job
        self.node = node
        self.assigned_ticks = assigned_ticks
        self.punishment = self.get_transfer_punishment()
        self.total_ticks = self.assigned_ticks + self.punishment

        self.status = "processing" # processing, finished, killed

    def get_transfer_punishment(self):
        """
        If the data required for the task is not saved on node, a punishment will be added.
        Punishments are found at the job_types section
        """
        punishment = 0
        if self.job.job_type not in self.node.meta_data:
            punishment = self.job.punishment
        return punishment

class Node:
    """
    A node can be seen as a machine in a distributed system. Nodes are assigned tasks to perform and
    keep track of the job and task it is currently employed to. Nodes can be assigned, killed (destroying 
    the progress made), free'd up and perform its work by crunching the task at hand.
    """
    def __init__(self, scheduler, id):
        """
        Initialize the node class with its parent scheduler and the ID that refers to itself in scheduler.nodes

        Inputs: scheduler (Scheduler class), id (int)
        """
        self.id = id
        self.scheduler = scheduler
        self.status = "accessible" # assigned, finished, killed, accessible
        self.task = None
        self.job = None
        self.meta_data = []

    def assign(self, task, job):
        """
        When a node gets a task assigned, the node has to be flagged as busy through the scheduler busy_nodes list.
        The supplied task and job are bound to the node.

        Input: task (Task class), job (Job class)
        """
        self.scheduler.free_nodes.remove(self.id)
        self.scheduler.busy_nodes.append(self.id)
        self.status = "assigned"
        self.task = task
        self.job = job

        # Make sure to not only bind the task to the node, but also the node to the task
        self.task.node = self
        
    def kill(self):
        """
        Kill and redistribute
        """
        if self.task != None:
            ticks2redistribute = self.task.assigned_ticks
        
            if self in self.job.assigned_nodes_ids:
                self.job.assigned_nodes_ids.remove(self)
            self.job.redistribute_ticks(ticks2redistribute)
        self.free_up()

    def free_up(self):
        """
        Freeing up a node ensures it becomes available to other tasks and jobs.
        This function "resets" a node to become accessible.
        """
        self.scheduler.busy_nodes.remove(self.id)
        self.scheduler.free_nodes.append(self.id)
        self.status = "accessible"
        self.job.assigned_nodes_ids.remove(self.id)
        self.job = None
        self.task = None

    def perform_work(self):
        """
        When called, this function ensures that the task's ticks are negatively incremented.
        When the node has finished the task (e.g. reached tick 0), it will mark itself as finished and
        ensures that the work done is subtracted from the job.
        """
        if self.task != None:
            self.task.total_ticks -= 1
            self.task.total_ticks = max(self.task.total_ticks, 0)
            if self.task.total_ticks == 0 and self.status != "finished":
                self.status = "finished"
                self.job.total_ticks -= self.task.assigned_ticks

class Job:
    """
    The Job class is a parent of tasks. The job represents a higher level task or problem that requires
    solving by distributing itself through tasks. Jobs are called at certain steps planned by the forecaster.
    The different job_types and their specifications are found at the top of this file. When creating tasks
    and assigning them to nodes, jobs divide their workload over the task and distribute it. Jobs keep track
    of the nodes they are assigned and of the work that is finished, thus ensuring that when a job is done,
    all ticks are crunched.
    """
    def __init__(self, scheduler, job_type):
        """
        Initialize the job based on the job type and calculate the number of ticks this job requires.
        The number of ticks is determined by a normal distribution.

        Inputs: scheduler (Scheduler class), job_type (str)
        """
        self.job_type = job_type
        self.scheduler = scheduler

        # Deepcopy is required as we want the start step not to change
        self.tick_queued = copy.deepcopy(self.scheduler.step)
        self.status = "queued" # queued, assigned, finished

        # Generate number of ticks the job requires
        self.total_ticks = round(abs(np.random.normal(
            loc=job_types[job_type][1],
            scale=job_types[job_type][2]
            )))
        self.job_specific_ticks = copy.deepcopy(self.total_ticks)

        self.punishment = job_types[job_type][3]
        self.assigned_nodes_ids = [] 

        # This is for delay scheduling only
        self.skip_count = 0

    def assign_nodes(self, nodes_ids):
        """
        When nodes are assigned to a job by the scheduler, it receives the ID's of the assigned nodes.
        Tasks are automatically created and the workload is evenly distributed.

        Inputs: nodes_ids (list of integers)
        """

        # We need the deepcopy here as nodes_ids is a pointer to the free nodes list
        # As this list changes in the enumerate loop below, we need a deepcopy to ensure
        # all nodes are enumerated over
        nodes_ids = copy.deepcopy(nodes_ids)
        self.status = "assigned"
        self.tick_assigned = copy.deepcopy(self.scheduler.step)
        n_nodes = len(nodes_ids)

        mean_ticks = int(self.total_ticks / n_nodes)
        leftover_ticks = self.total_ticks % n_nodes

        # Distribute ticks over nodes and tasks
        for i, n in enumerate(nodes_ids):
            self.assigned_nodes_ids.append(n)
            if i < leftover_ticks:
                # Add an extra leftover tick to ensure all ticks are distrbuted
                self.scheduler.nodes[n].assign(Task(self, mean_ticks + 1, self.scheduler.nodes[n]), self)
            else:
                self.scheduler.nodes[n].assign(Task(self, mean_ticks, self.scheduler.nodes[n]), self)

    def check_finished(self):
        """
        Jobs check each tick if they are finished by summing the remaining ticks of the
        nodes assigned to it. When it is sure it is finished by checking if its own ticks
        and those of the nodes are 0, the is marked as finished and frees up the assigned nodes.
        """
        node_tick_sum = 0

        for n in self.assigned_nodes_ids:
            node_tick_sum += self.scheduler.nodes[n].task.total_ticks

        if self.total_ticks == 0 and node_tick_sum == 0:
            self.status = "finished"
            self.tick_finished = copy.deepcopy(self.scheduler.step)
            self.scheduler.active_jobs.remove(self)
            self.scheduler.finished_jobs.append(self)
            for n in copy.deepcopy(self.assigned_nodes_ids):
                self.scheduler.nodes[n].free_up()

            self.optimal_time = math.ceil(self.job_specific_ticks / n_nodes)
            self.total_time_ratio = (self.tick_finished - self.tick_queued) / self.optimal_time
            self.processing_time_ratio = (self.tick_finished - self.tick_assigned) / self.optimal_time
            self.waiting_time_ratio = (self.tick_assigned - self.tick_queued) / self.optimal_time

            self.scheduler.total_ratios.append(self.total_time_ratio)
            self.scheduler.processing_ratios.append(self.processing_time_ratio)
            self.scheduler.waiting_ratios.append(self.waiting_time_ratio)
            self.scheduler.total_ticks_waited += (self.tick_assigned - self.tick_queued)

    def redistribute_ticks(self, ticks2redistribute):
        mean_ticks = int(ticks2redistribute / len(self.assigned_nodes_ids))
        leftover_ticks = ticks2redistribute % len(self.assigned_nodes_ids)
        for i in self.assigned_nodes_ids:
            if i < leftover_ticks:
                self.scheduler.nodes[i].task.assigned_ticks += mean_ticks + 1
                self.scheduler.nodes[i].task.total_ticks += mean_ticks + 1
            else:
                self.scheduler.nodes[i].task.assigned_ticks += mean_ticks
                self.scheduler.nodes[i].task.total_ticks += mean_ticks

class Scheduler:
    """
    The scheduler is the largest overarching object in our setup. EXPAND
    """
    def __init__(self):
        """
        Initialize scheduler by creating a forecast of jobs and creating the nodes
        """
        self.forecaster = Forecaster(self)
        self.future_jobs = self.forecaster.forecast_future_jobs()

        self.step = 0
        self.queue = []
        self.active_jobs = []
        self.finished_jobs = []

        self.nodes = {}
        self.free_nodes = []
        self.busy_nodes = []
        
        for i in range(n_nodes):
            self.nodes[i] = Node(self, i)
            self.free_nodes.append(i)
        
        self.generate_data_splits()
        self.distribute_data()

        # These are used for calculating the performance
        self.total_ratios = []
        self.processing_ratios = []
        self.waiting_ratios = []
        self.total_ticks_waited = 0

    def start_simulation(self):
        """
        This function triggers the simulation of the distributed environment. The different elements
        per step in the loop below are explained separately.
        """
        for step in range(experiment_duration):
            self.step = step

            # 1. Check for finished jobs and free up nodes
            self.check_finished_jobs()
            # 3. Check for new jobs for the queue
            self.update_queue()
            # 4. Schedule due jobs
            self.schedule()
            # 5. Trigger node ticks and queue waiting ticks
            self.trigger_tick()
            
            # Print the queue at an interval to monitor queue length
            if step % 100 == 0 and show_progress:
                print("At step:", step, "\tQueue length:", len(self.queue), "\tActive jobs:", len(self.active_jobs), "\t\tN active nodes", len(self.busy_nodes))

        if run_until_finished:
            # This is not recommended and is not used in our experiments
            while len(self.queue) != 0 or len(self.active_jobs) != 0:
                self.step += 1
                step = self.step
                self.check_finished_jobs()
                self.update_queue()
                self.schedule()
                self.trigger_tick()
                if step % 100 == 0 and show_progress:
                    print("At step:", step, "\tQueue length:", len(self.queue), "\tActive jobs:", len(self.active_jobs), "\t\tN active nodes", len(self.busy_nodes))
            
        # Print results
        print("Results for scheduler:\t", scheduler_type)
        print("Mean total ratio:\t", np.mean(self.total_ratios))
        print("Mean processing ratio:\t", np.mean(self.processing_ratios))
        print("Mean waiting ratio:\t", np.mean(self.waiting_ratios))
        print("Total ticks waited:\t", f'{self.total_ticks_waited:,}')
        print("Total number of jobs processed:\t", f'{len(self.total_ratios)}')

    def generate_data_splits(self):
        """
        TODO Finish documentation
        This function generates the different splits of nodes that can then be used to distribute data. A split in
        the simulation is similar to how a rack IRL functions.

        This code was written while on a trainride and is subject to change as it is horrible
        """
        split_sizes = []
        self.splits = []
        mean_nodes_per_split = int(n_nodes / n_data_splits)
        leftover_splits = n_nodes % n_data_splits
        for i in range(n_data_splits):
            if i < leftover_splits:
                split_sizes.append(mean_nodes_per_split + 1)
            else:
                split_sizes.append(mean_nodes_per_split)   
        i = 0
        for s in split_sizes:
            self.splits.append(list(range(i, i + s)))
            i += s
        self.splits.append(self.splits.pop(0))

    def distribute_data(self):
        """
        TODO Finish documentation
        Distribute the data over generated splits of nodes such that the data for each task is atleast
        store on n_data_copies nodes.

        This code was written while on a trainride and is subject to change as it is horrible
        """
        job_keys = list(job_types.keys())
        np.random.shuffle(job_keys) # Shuffle jobs so the presented order becomes irrelevant
        for job in job_keys:
            for s in self.splits[0:n_data_copies]:
                for n in s:
                    self.nodes[n].meta_data.append(job)
            self.splits.append(self.splits.pop(0))

    def check_finished_jobs(self):
        """
        Calls all active jobs to check whether they are finished or not. 
        Sometimes, jobs can get stuck or will take too long, we remove them and kill their nodes after a waiting period.
        """
        for j in self.active_jobs:
            j.check_finished()
            # if self.step - j.tick_queued > 10000:
            #     for n in j.assigned_nodes_ids:
            #         self.nodes[n].kill()
            #     self.active_jobs.remove(j)

    def update_queue(self):
        """
        Checks if the current step occurs in the forecast of jobs. If this is the case, create the job and add
        it to the top of the queue.
        """
        if self.step in self.future_jobs:
            self.queue.append(Job(self, self.future_jobs[self.step]))
    
    def trigger_tick(self):
        """
        This function ensures that all nodes perform their work exactly 1 tick per step
        """
        for n in self.nodes:
            self.nodes[n].perform_work()
    
    def schedule(self):
        """
        This is a delegating function, that allows for easy selection of a scheduler type.
        Please refer to the actual called functions for an explanation per type.
        """
        if len(self.queue) != 0:
            if scheduler_type == "fifo":
                self.fifo()

            elif scheduler_type == "fair":
                self.fair()

            elif scheduler_type == "delay_fair":
                self.delay_fair()

    def fifo(self):
        """
        First in first out scheduler. Currently simply deploys the oldest job in the queue when
        all nodes are free. If not, it waits until the current job is fully(!) finished.    
        
        In the future, this be expanded to also perform partial assignment and redistribution of nodes.
        """
        if len(self.free_nodes) == len(self.nodes):
            task2assign = self.queue.pop(0)
            self.active_jobs.append(task2assign)
            self.active_jobs[-1].assign_nodes(self.free_nodes)
    
    def fair(self):
        """
        Naive fair sharing scheduler, shares the resources available over all jobs equally.
        The minimum amount of resources allocated is 1 node per job. If all nodes are at the minimal
        allocation, no other job can be scheduled
        """
        if len(self.active_jobs) < n_nodes:
            job2assign = self.queue.pop(0)
            self.fair_redistribute(job2assign)

    def get_node_to_reassign(self, selected_job_index, target_job_type):
        """
        Select a node of a job to free up based on the target job type. 
        If the job contains a node with the target job type, return it. 
        Else, return a random node
        """
        for n in self.active_jobs[selected_job_index].assigned_nodes_ids:
            if target_job_type in self.nodes[n].meta_data:
                return n
        return np.random.choice(self.active_jobs[selected_job_index].assigned_nodes_ids)

    def fair_redistribute(self, job2assign):
        """
        This function frees up nodes in naive fair sharing and assigns them such that all running jobs 
        have (almost) equal resources.
        Priority is given to freeing up nodes that contain the job type data
        """
        # Make sure that the nodes with the most assigned nodes also give up the most nodes
        if len(self.active_jobs) > 0:
            n_nodes4new_job = int(n_nodes / (len(self.active_jobs) + 1)) - len(self.free_nodes)
            for i in range(n_nodes4new_job):
                n_nodes_per_job = []
                for j in self.active_jobs:
                    n_nodes_per_job.append(len(j.assigned_nodes_ids))
                selected_job_index = np.argmax(n_nodes_per_job)
                node2free_id = self.get_node_to_reassign(selected_job_index, job2assign.job_type)
                self.nodes[node2free_id].kill()

        self.active_jobs.append(job2assign)
        self.active_jobs[-1].assign_nodes(self.free_nodes)

    def delay_fair(self):
        """
        Simple delay fair scheduling (not HFS) with data locality.

        We tried to follow the pseudocode as much as possible, but ended up deviating quite a bit
        to get the same effect.
        """

        if len(self.active_jobs) < n_nodes:
            # Do not schedule a new job if all running jobs already have the minimum resources of 1 node
            if self.queue[0].skip_count >= skip_count_max:
                # The job has been skipped enough and should immediately be executed
                job2assign = self.queue.pop(0)
                self.fair_redistribute(job2assign)

            else:
                # The next job in the queue can be postponed if necessary
                jobs2remove = [] # Jobs cannot be removed during looping, this list will be used to remove them afterwards
                for j in self.queue:
                    # Loop through queue from old to new jobs
                    if len(self.free_nodes) != 0:
                        # We still have nodes that can be scheduled
                        nodes2assign = []
                        for n in self.free_nodes:
                            # Only schedule jobs on nodes that are near their data
                            if j.job_type in self.nodes[n].meta_data:
                                nodes2assign.append(n)

                        if len(nodes2assign) > 0:
                            # We found nodes to be assigned to the job
                            jobs2remove.append(j)
                            self.active_jobs.append(j)
                            j.assign_nodes(nodes2assign)
                        else:
                            # Increment skip count of job cant be scheduled yet
                            j.skip_count += 1

                for j in jobs2remove:
                    # Remove the scheduled jobs from the queue
                    self.queue.remove(j)

# Create an instance of the scheduler and start simulating
simple_scheduler = Scheduler()
simple_scheduler.start_simulation()
