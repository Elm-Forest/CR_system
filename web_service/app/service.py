import asyncio
import base64
import os

from werkzeug.utils import secure_filename

from exe import delete_source, delete_outputs
from utils.common import *


def upload_service(cloudy_image, sar_image, target_image):
    cloudy_image_name = secure_filename(cloudy_image.filename)
    cloudy_image_path = os.path.join(os.path.join(SOURCE_DIR, 's2'), cloudy_image_name)
    sar_image_path = os.path.join(os.path.join(SOURCE_DIR, 's1'), cloudy_image_name)
    target_image_path = os.path.join(os.path.join(SOURCE_DIR, 'target'), cloudy_image_name)
    cloudy_image.save(cloudy_image_path)
    sar_image.save(sar_image_path)
    target_image.save(target_image_path)
    return cloudy_image_name


def preprocess_service(image_name, cr_exe):
    cr_exe.preprocess_images(image_name)
    return {
        'cloudy_image': encode_file_to_base64(
            os.path.join(DEFAULT_META_OUTPUT_DIR, image_name + CLOUDY_POSTFIX + OUTPUT_IMAGES_POSTFIX)),
        'target_image': encode_file_to_base64(
            os.path.join(DEFAULT_META_OUTPUT_DIR, image_name + TARGET_POSTFIX + OUTPUT_IMAGES_POSTFIX)),
        'sar_image': encode_file_to_base64(
            os.path.join(DEFAULT_META_OUTPUT_DIR, image_name + SAR_POSTFIX + OUTPUT_IMAGES_POSTFIX)),
        'mask_image': encode_file_to_base64(
            os.path.join(DEFAULT_META_OUTPUT_DIR, image_name + MASK_POSTFIX + OUTPUT_IMAGES_POSTFIX))
    }


async def async_predict_dsen(image_name, cr_exe):
    cr_exe.predict_dsen(image_name)
    print("dsen done")


async def async_predict_glf(image_name, cr_exe):
    cr_exe.predict_glf(image_name)
    print("glf done")


async def async_predict_uncrtain(image_name, cr_exe):
    cr_exe.predict_uncrtain(image_name)
    print("uncertain done")


def infer_service(image_name, cr_exe):
    async def run_inference():
        await asyncio.gather(
            async_predict_dsen(image_name, cr_exe),
            async_predict_glf(image_name, cr_exe),
            async_predict_uncrtain(image_name, cr_exe)
        )

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(run_inference())

    out4 = cr_exe.postprocess_images(image_name)
    out5 = cr_exe.predict_meta(image_name)
    outputs = {**out4, **out5}
    outputs["input1_image"] = encode_file_to_base64(outputs["input1_dir"])
    outputs["input2_image"] = encode_file_to_base64(outputs["input2_dir"])
    outputs["input3_image"] = encode_file_to_base64(outputs["input3_dir"])
    outputs["output_image"] = encode_file_to_base64(outputs["image_dir"])
    outputs["mask_image"] = encode_file_to_base64(outputs["mask_dir"])
    del outputs["input1_dir"]
    del outputs["input2_dir"]
    del outputs["input3_dir"]
    del outputs["image_dir"]
    del outputs["mask_dir"]
    delete_source(image_name)
    delete_outputs(image_name)
    return outputs


def encode_file_to_base64(filepath):
    with open(filepath, 'rb') as file:
        encoded_content = base64.b64encode(file.read()).decode('utf-8')
    return encoded_content
