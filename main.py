#!/usr/bin/env python

import enhance

# enhance.fix_all_from_mongo_and_update(remote_update=True, local_output=True, first_style_index=211, first_image_index=20)
enhance.fix_all_from_mongo_and_update(remote_update=False, local_output=True, style_limit=10, images_per_style=10)
