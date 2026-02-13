import datetime as dt
import shutil

import modal

from zeus.inference.inference_flows import GeoFlows
from zeus.infrastructure.modal.failures import RETRIES_WINDFLOW, TIMEOUT_WINDFLOW
from zeus.infrastructure.modal.images import WINDFLOW_IMAGE
from zeus.infrastructure.modal.reservations import (
    CPU_WINDFLOW,
    MEMORY_WINDFLOW,
)
from zeus.infrastructure.modal.secrets import SECRETS
from zeus.infrastructure.modal.volumes import VOLUMES

LAG = 12
MODEL_CKPT_PATH = "/windflow/model_weights/windflow.raft.pth.tar"
CACHE_DIR = "/windflow/cache"


app = modal.App("windflow_live")


@app.function(
    image=WINDFLOW_IMAGE,
    # TODO: improve this.
    schedule=modal.Cron(f"{LAG} * * * *"),
    volumes=VOLUMES,
    secrets=SECRETS,
    retries=RETRIES_WINDFLOW,
    cpu=CPU_WINDFLOW,
    memory=MEMORY_WINDFLOW,
    timeout=TIMEOUT_WINDFLOW,
)
def run_windflow():
    inference = GeoFlows(
        sat="goes19", model_ckpt_path=MODEL_CKPT_PATH, cache_dir=CACHE_DIR
    )
    now = dt.datetime.now().replace(second=0, microsecond=0)
    t2 = now - dt.timedelta(minutes=LAG)
    t2 = t2 - dt.timedelta(minutes=t2.minute % 10)
    inference.flows_by_time(t2)
    shutil.rmtree(CACHE_DIR)
