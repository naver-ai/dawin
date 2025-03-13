'''
official checkpoints provided in the codebase of "Editing Models with Task Arithmetic (ICLR23)" 
'''
import gdown
import os
prefix='/YOUR_ROOT_PATH/dawin/dawin_mtl/checkpoints/ViT-B-32'

ftdict={
    'GTSRB':'1BDEpJSL0GWGfD93OvymW_oZZeWfyYclM',
    'SUN397':'1HWs-n0GEewLQ9hXnOJYtm85GtJpJl35Q',
    'EuroSAT':'1EZIkpSox2tuXRDADFw2Ih9Rfown94BKU',
    'MNIST':'1IYE99VnUcKso2UOUYRfuts8MtrZ7IdqU',
    'Cars':'101U8_jLvsDg6WePgsDK9QvEFw1_9jnbC',
    'DTD':'1octqDdrX8vOSRfMWKHSBodPiC5OS4XBA',
    'RESISC45':'1Glu2Hky3qa58LJgtWhwDa50CLkCPdkuo',
    'SVHN':'1_xM_tfAJPm_0YoonnaSn4rVeek2qrzk9'
}

headdict={
    'GTSRB':'1j2yVXOGdzHzcT6Zp5kc8xold77VRxwIC',
    'SUN397':'1kXAWZZ9P4p_fsqQ8UjOKEWGViksF8rsx',
    'EuroSAT':'1-aqbjz5rKJDPPtduQk1L7n0OimWkpRM6',
    'MNIST':'1_9lBNw877gb9fGcXQTZljVANgU8e_9Z4',
    'Cars':'1NZmJHeELIwGDa0vhoPHp89J4iXpsVhb5',
    'DTD':'1hWP7hN_inUnuZIin_4yooD74iKQfNq5q',
    'RESISC45':'1HrUa1eDtG_CFd_TYox-tO_8BHqME1YR3',
    'SVHN':'1AXWsTQjk5KKaRsIJK3wszZT4RvEWJ3cM'
}

for ds, fid in ftdict.items():
    print(f'start download {ds} to {prefix}')
    if not os.path.exists(prefix+f'/{ds}'): os.makedirs(prefix+f'/{ds}',exist_ok=True)
    output = prefix + f'/{ds}/finetuned.pt'
    #url = f'https://drive.google.com/file/d/{fid}/view?usp=drive_link'
    url = f'https://drive.google.com/uc?id={fid}'
    gdown.download(url, output, quiet=False)

for ds, fid in headdict.items():
    print(f'start download {ds} head to {prefix}')
    output = prefix + f'/head_{ds}.pt'
    #url = f'https://drive.google.com/file/d/{fid}/view?usp=drive_link'
    url = f'https://drive.google.com/uc?id={fid}'
    gdown.download(url, output, quiet=False)

print('start download zeroshot.pt')
output = prefix + f'/zeroshot.pt'
fid='145ZjznF8HyTQvtlK1Mw9ybArVhm4b-8C'
url = f'https://drive.google.com/uc?id={fid}'
gdown.download(url, output, quiet=False)
