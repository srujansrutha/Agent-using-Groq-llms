import boto3
import os
import json
from datetime import datetime, timedelta, timezone
from langchain_community.tools import Tool
from langchain_groq import ChatGroq
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage
from langchain.memory import ConversationBufferMemory

# Configuration
os.environ["GROQ_API_KEY"] = "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"  # Replace with your actual key

# Initialize AWS clients with error handling
try:
    ce = boto3.client('ce')
    cloudwatch = boto3.client('cloudwatch')
    ec2 = boto3.client('ec2')
except Exception as e:
    print(f"AWS Client initialization failed: {str(e)}")
    exit(1)

# -------------------------------
# Helper Functions
# -------------------------------

def get_aws_cost(*args, **kwargs) -> list:
    """Retrieve AWS cost data for the last 12 months."""
    try:
        end_date = datetime.now(timezone.utc).date()
        start_date = end_date - timedelta(days=365)
        response = ce.get_cost_and_usage(
            TimePeriod={'Start': start_date.isoformat(), 'End': end_date.isoformat()},
            Granularity='MONTHLY',
            Metrics=['BlendedCost']
        )
        return [{
            "month": item["TimePeriod"]["Start"],
            "cost": round(float(item["Total"]["BlendedCost"]["Amount"]), 2)
        } for item in response.get("ResultsByTime", [])]
    except Exception as e:
        return [{"error": f"Cost retrieval failed: {str(e)}"}]

def get_ec2_utilization_yearly(instance_id: str) -> dict:
    """
    Get daily EC2 CPU utilization data for the past year.
    Returns a list of dictionaries, each containing:
      - date (ISO string)
      - average_cpu: average CPU utilization for that day
      - spike_count: count of datapoints above 60%
    """
    try:
        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(days=365)
        # Get daily datapoints: using period = 86400 (one day)
        response = cloudwatch.get_metric_statistics(
            Namespace='AWS/EC2',
            MetricName='CPUUtilization',
            Dimensions=[{'Name': 'InstanceId', 'Value': instance_id}],
            StartTime=start_time,
            EndTime=end_time,
            Period=86400,
            Statistics=['Average'],
            Unit='Percent'
        )
        datapoints = response.get('Datapoints', [])
        # Organize datapoints by day
        daily_data = {}
        for dp in datapoints:
            day = dp['Timestamp'].date().isoformat()
            if day not in daily_data:
                daily_data[day] = []
            daily_data[day].append(dp['Average'])
        
        daily_util = []
        for day, values in daily_data.items():
            avg_cpu = round(sum(values) / len(values), 2)
            # Count spikes where utilization exceeds 60%
            spike_count = sum(1 for v in values if v > 60)
            daily_util.append({
                "date": day,
                "average_cpu": avg_cpu,
                "spike_count": spike_count
            })
        # Sort by date ascending
        daily_util.sort(key=lambda x: x["date"])
        return {"instance_id": instance_id, "daily_utilization": daily_util}
    except Exception as e:
        print(f"Utilization error for {instance_id}: {str(e)}")
        return {"instance_id": instance_id, "error": str(e)}

def get_ec2_instances(*args, **kwargs) -> list:
    """Retrieve running EC2 instances along with yearly CPU utilization data."""
    try:
        instances = []
        paginator = ec2.get_paginator('describe_instances')
        for page in paginator.paginate():
            for reservation in page['Reservations']:
                for instance in reservation['Instances']:
                    if instance['State']['Name'] == 'running':
                        utilization_data = get_ec2_utilization_yearly(instance['InstanceId'])
                        instances.append({
                            "id": instance['InstanceId'],
                            "type": instance['InstanceType'],
                            "yearly_cpu_utilization": utilization_data
                        })
        return instances
    except Exception as e:
        return [{"error": f"Instance retrieval failed: {str(e)}"}]

def get_ebs_snapshots_analysis(*args, **kwargs) -> list:
    """
    Retrieve EBS snapshots that are older than 6 months and are not attached to any EC2 instance.
    """
    try:
        snapshots = []
        # Get snapshots owned by self
        response = ec2.describe_snapshots(OwnerIds=['self'])
        now = datetime.now(timezone.utc)
        six_months_ago = now - timedelta(days=180)
        for snap in response.get("Snapshots", []):
            created_time = snap["StartTime"]
            if created_time < six_months_ago:
                volume_id = snap.get("VolumeId")
                # Check if volume is attached to any instance
                attached = False
                if volume_id:
                    vol_response = ec2.describe_volumes(Filters=[{'Name': 'volume-id', 'Values': [volume_id]}])
                    for vol in vol_response.get("Volumes", []):
                        if vol.get("Attachments"):
                            attached = True
                            break
                if not attached:
                    snapshots.append({
                        "snapshot_id": snap["SnapshotId"],
                        "volume_id": volume_id,
                        "start_time": created_time.isoformat(),
                        "description": snap.get("Description", "")
                    })
        return snapshots
    except Exception as e:
        return [{"error": f"Snapshot analysis failed: {str(e)}"}]

# -------------------------------
# Agent configuration
# -------------------------------

def create_agent() -> AgentExecutor:
    """Create and configure the multi-agent LangChain agent with added tools."""
    tools = [
        Tool(
            name="AWS_Cost_Analyzer",
            func=get_aws_cost,
            description="Retrieves AWS cost data for the last 12 months."
        ),
        Tool(
            name="EC2_Instance_Inspector",
            func=get_ec2_instances,
            description="Fetches running EC2 instances along with yearly CPU utilization metrics (daily averages and spike counts)."
        ),
        Tool(
            name="EBS_Snapshot_Analyzer",
            func=get_ebs_snapshots_analysis,
            description="Fetches EBS snapshots older than 6 months that are not attached to any EC2 instance."
        )
    ]

    llm = ChatGroq(
        temperature=0.2,
        model_name="llama-3.3-70b-versatile",  # Example model name
        max_tokens=4096
    )

    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=(
            "You are an AWS FinOps and Cloud Optimization Expert. Follow these steps:\n"
            "1. Use AWS_Cost_Analyzer to retrieve cost data for the past 12 months.\n"
            "2. Use EC2_Instance_Inspector to fetch running EC2 instances along with detailed yearly CPU metrics (daily average and count of times above 60%).\n"
            "3. Use EBS_Snapshot_Analyzer to identify snapshots older than 6 months that are not attached to any instance.\n"
            "4. Based on this data, generate actionable recommendations to optimize costs and propose downsizing actions if applicable.\n"
            "Present the results in clear, organized sections with markdown formatting."
        )),
        HumanMessage(content="{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])

    agent = create_tool_calling_agent(llm, tools, prompt)
    
    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        memory=ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        ),
        handle_parsing_errors=True,
        max_iterations=7,  # Allow more iterations for multi-step analysis
        early_stopping_method="force"
    )

# -------------------------------
# Main execution
# -------------------------------
if __name__ == "__main__":
    # Validate environment
    if not os.getenv("GROQ_API_KEY"):
        print("ERROR: GROQ_API_KEY environment variable not set")
        exit(1)
        
    try:
        agent = create_agent()
        print("\n🔹 AWS FinOps Optimization Analyzer 🔹\n")
        
        response = agent.invoke({
            "input": "Provide a comprehensive analysis of our AWS infrastructure, including cost data, EC2 instance performance (daily CPU average and spike counts over the past year), and EBS snapshot usage. Based on this, generate downsizing and optimization recommendations.",
            "chat_history": []
        })
        
        print("\n📊 Analysis and Recommendations:\n")
        print(response["output"])
        
    except Exception as e:
        print(f"\n❌ Critical Error: {str(e)}")
        print("Troubleshooting Steps:")
        print("1. Verify AWS credentials have permissions for Cost Explorer, CloudWatch, EC2, and EBS snapshot access")
        print("2. Check GROQ API key validity")
        print("3. Ensure AWS resources exist in the configured region")
        print("4. Test individual functions separately")
