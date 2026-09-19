"""JOIN4b section repair: weight-bucket profile on the LOCKED config.

v2 finding: cb-name sections leaked ~110ms/tok into MISC (shexp matmuls
run before their "ffn_shexp" opener; TAIL markers never match). This
kernel adds weight-bucket timers (EVERY matmul by loader weight name:
immutable) + a nodelist ground-truth dump, and re-profiles resident +
locked b6 on 12 spread prompts. Arm comparison (tps/RSS/traffic) stands
from v2; ONLY the component split is repaired here.

Same binary/flags/seeds/N_GEN as JOIN4; arms resident x2 + b6 x2 on
PIDS (12 spread prompts). Nodelist ground truth on one short run.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import threading
import time
from pathlib import Path

WORK = Path("/kaggle/working")
SCRATCH = Path("/tmp/native-sparse-edge0join4b-v1")
OUT = WORK / "native-sparse-edge0join4b-v1-results"
PIDS = [0, 1, 4, 6, 7, 10, 12, 13, 15, 17, 18, 21]
LLAMA = SCRATCH / "llama.cpp"
BUILD = LLAMA / "build-native"
CLI = BUILD / "bin" / "llama-cli"
QUANTIZE = BUILD / "bin" / "llama-quantize"

LLAMA_COMMIT = "3057bb66c86c46d5781e50e85462a760ba7d1feb"
BASE = "3.6"
MODELS = {
    "3.6": {
        "repo": "a483e9e6cbd595906af30beda3187c2663a1118c",
        "file": "Qwen3.6-35B-A3B-UD-IQ2_XXS.gguf",
        "size": 10_756_586_464,
        "sha": "2e8f5f705355c56311432d0a8a5d14a696dbb7e4b197d05c75ba805fc1857bef",
        "hf": "unsloth/Qwen3.6-35B-A3B-GGUF",
    },
}
Q2K_NAME = "Qwen3.6-35B-A3B-UD-Q2K-experts.gguf"
K1, K2 = 4, 16
THREADS = 4
N_GEN = 80
SETTLE_SEC = 10

PROMPTS = [
    ("medical", "A 3-year-old has watery diarrhoea and is thirsty. Explain how to assess dehydration and what home treatment to start."),
    ("medical", "A pregnant woman has headache, swollen face, and high blood pressure. What danger signs and referral steps apply?"),
    ("medical", "List the steps to prepare oral rehydration solution at home and when to seek care."),
    ("medical", "A newborn feels cold and feeds poorly. Describe warming, feeding, and referral actions."),
    ("medical", "Explain why deworming tablets are given at school and what side effects to watch for."),
    ("medical", "A child has fast breathing and chest indrawing. Outline assessment and urgent actions."),
    ("general", "Explain how rain forms and why some seasons are drier than others."),
    ("general", "Describe how a market day works in a small town, from morning to evening."),
    ("general", "Summarize the plot of a story about a fisherman who finds something unexpected."),
    ("general", "Compare travelling by bus and by motorcycle taxi for a long trip."),
    ("general", "Explain what a budget is and how a family can make one."),
    ("general", "Describe the steps to plant maize from land preparation to harvest."),
    ("swahili", "Eleza kwa kifupi kwa nini maji ya kunywa yenye chumvi na sukari husaidia mtoto mwenye kuharisha."),
    ("swahili", "Ni dalili zipi za hatari zinahitaji mtoto apelekwe hospitali haraka?"),
    ("swahili", "Andika mpango mfupi wa usafi wa mikono kwa shule ya kijijini."),
    ("swahili", "Tofautisha ukweli, maoni, na dhana kwa lugha rahisi ya Kiswahili."),
    ("swahili", "Mweleze mgonjwa kwa heshima kwa nini anatakiwa kufuata dozi aliyopewa."),
    ("instruction", "Answer in exactly three bullet points: how to prepare for a job interview."),
    ("instruction", "Give a two-sentence answer and one caveat about using online medical advice."),
    ("instruction", "Create a small table comparing prevention, diagnosis, and treatment."),
    ("instruction", "Ask one useful clarification question before answering an underspecified request."),
    ("short", "Hello."),
    ("short", "What should I do next?"),
]

ARMS = [
    {"name": "resident", "bounded": False, "reps": 2},
    {"name": "bounded_6gb", "bounded": True, "slots": 3662, "pins": "6.0", "reps": 2},
]

# Embedded pin sets (one key per line), injected at kernel build from
# cache_config_k4.json (LOCKED; transfer-validated).
PINS_3 = '175\n238\n411\n501\n561\n729\n967\n1004\n1178\n1216\n1438\n1468\n1615\n1674\n1827\n2021\n2109\n2115\n2320\n2501\n2618\n2756\n2927\n2976\n3111\n3265\n3397\n3482\n3636\n3751\n3900\n4005\n4266\n4348\n4416\n4467\n4621\n4800\n4929\n5092\n5176\n5187\n5392\n5423\n5690\n5879\n5995\n5999\n6183\n6289\n6413\n6469\n6708\n6793\n7074\n7145\n7320\n7338\n7506\n7574\n7835\n7872\n8001\n8165\n8273\n8370\n8488\n8522\n8705\n8820\n8992\n9048\n9345\n9414\n9530\n9612\n9820\n9848\n10146\n10173\n'
PINS_4 = '175\n238\n411\n501\n561\n729\n967\n1004\n1178\n1216\n1438\n1468\n1615\n1674\n1827\n2021\n2109\n2115\n2320\n2501\n2618\n2756\n2927\n2976\n3111\n3265\n3397\n3482\n3636\n3751\n3900\n4005\n4266\n4348\n4416\n4467\n4621\n4800\n4929\n5092\n5176\n5187\n5392\n5423\n5690\n5879\n5995\n5999\n6183\n6289\n6413\n6469\n6708\n6793\n7074\n7145\n7320\n7338\n7506\n7574\n7835\n7872\n8001\n8165\n8273\n8370\n8488\n8522\n8705\n8820\n8992\n9048\n9345\n9414\n9530\n9612\n9820\n9848\n10146\n10173\n'
PINS_5 = '2\n13\n14\n35\n38\n43\n54\n78\n81\n93\n95\n101\n112\n115\n124\n134\n136\n140\n141\n151\n167\n175\n186\n197\n199\n200\n209\n210\n222\n224\n235\n238\n267\n272\n283\n302\n303\n305\n318\n353\n368\n380\n383\n396\n402\n409\n411\n437\n438\n451\n454\n473\n489\n490\n501\n506\n507\n510\n515\n523\n532\n553\n555\n561\n564\n572\n584\n588\n596\n602\n604\n618\n621\n632\n633\n662\n665\n670\n676\n679\n700\n705\n722\n729\n740\n765\n780\n781\n784\n787\n793\n794\n804\n805\n806\n812\n874\n885\n887\n916\n923\n927\n934\n936\n945\n955\n956\n965\n967\n979\n982\n983\n993\n994\n998\n1001\n1003\n1004\n1038\n1039\n1042\n1044\n1068\n1075\n1076\n1078\n1082\n1091\n1098\n1112\n1118\n1122\n1126\n1161\n1166\n1168\n1173\n1178\n1193\n1202\n1216\n1217\n1219\n1224\n1233\n1245\n1246\n1259\n1265\n1277\n1282\n1296\n1320\n1323\n1328\n1341\n1343\n1345\n1357\n1361\n1370\n1372\n1390\n1395\n1405\n1415\n1431\n1438\n1445\n1454\n1466\n1468\n1473\n1489\n1493\n1498\n1508\n1515\n1516\n1530\n1537\n1550\n1585\n1588\n1590\n1595\n1596\n1601\n1615\n1618\n1620\n1644\n1651\n1652\n1664\n1674\n1679\n1685\n1686\n1693\n1703\n1718\n1732\n1751\n1762\n1763\n1782\n1786\n1790\n1806\n1815\n1821\n1824\n1827\n1828\n1830\n1868\n1874\n1891\n1909\n1938\n1961\n1963\n1965\n1968\n1973\n1983\n1989\n1991\n2009\n2021\n2024\n2052\n2053\n2081\n2082\n2083\n2085\n2091\n2094\n2104\n2105\n2109\n2111\n2114\n2115\n2118\n2137\n2146\n2152\n2155\n2158\n2164\n2205\n2207\n2212\n2214\n2221\n2228\n2231\n2232\n2239\n2268\n2270\n2272\n2282\n2285\n2287\n2306\n2308\n2312\n2320\n2328\n2333\n2348\n2351\n2358\n2363\n2379\n2383\n2398\n2406\n2420\n2424\n2430\n2436\n2456\n2460\n2465\n2471\n2475\n2484\n2495\n2501\n2502\n2507\n2546\n2568\n2569\n2572\n2575\n2591\n2617\n2618\n2623\n2625\n2628\n2645\n2648\n2651\n2669\n2677\n2679\n2682\n2690\n2693\n2706\n2717\n2725\n2729\n2734\n2745\n2747\n2756\n2758\n2794\n2799\n2803\n2807\n2808\n2819\n2828\n2836\n2843\n2852\n2853\n2855\n2858\n2875\n2880\n2894\n2901\n2916\n2922\n2923\n2924\n2927\n2929\n2944\n2952\n2961\n2969\n2970\n2973\n2976\n2990\n2997\n2998\n3005\n3022\n3055\n3056\n3058\n3071\n3075\n3088\n3111\n3115\n3117\n3122\n3129\n3132\n3133\n3137\n3141\n3146\n3165\n3179\n3187\n3193\n3195\n3200\n3217\n3219\n3232\n3238\n3265\n3266\n3271\n3280\n3282\n3292\n3299\n3301\n3331\n3341\n3347\n3357\n3361\n3371\n3372\n3375\n3394\n3397\n3416\n3417\n3430\n3440\n3441\n3450\n3461\n3462\n3477\n3482\n3503\n3510\n3512\n3518\n3527\n3557\n3575\n3577\n3612\n3620\n3635\n3636\n3645\n3648\n3655\n3666\n3670\n3682\n3683\n3687\n3690\n3709\n3715\n3721\n3731\n3741\n3743\n3744\n3751\n3756\n3773\n3776\n3790\n3805\n3816\n3836\n3844\n3847\n3876\n3894\n3898\n3900\n3911\n3928\n3945\n3950\n3955\n3983\n3986\n3988\n3991\n3992\n4002\n4003\n4005\n4006\n4012\n4017\n4026\n4042\n4044\n4049\n4053\n4056\n4060\n4069\n4073\n4076\n4079\n4082\n4090\n4101\n4122\n4124\n4126\n4128\n4138\n4142\n4147\n4148\n4149\n4155\n4170\n4178\n4191\n4198\n4212\n4226\n4230\n4246\n4252\n4259\n4266\n4277\n4286\n4301\n4318\n4325\n4329\n4330\n4336\n4339\n4348\n4349\n4352\n4357\n4367\n4369\n4372\n4375\n4383\n4387\n4412\n4416\n4430\n4434\n4439\n4462\n4467\n4490\n4494\n4506\n4510\n4518\n4537\n4541\n4542\n4546\n4550\n4583\n4588\n4592\n4593\n4594\n4600\n4602\n4609\n4613\n4621\n4624\n4636\n4651\n4655\n4665\n4675\n4688\n4694\n4724\n4734\n4736\n4738\n4742\n4758\n4763\n4776\n4778\n4785\n4786\n4787\n4800\n4805\n4809\n4811\n4828\n4834\n4839\n4855\n4858\n4859\n4868\n4869\n4871\n4876\n4878\n4884\n4894\n4899\n4904\n4906\n4907\n4912\n4915\n4922\n4929\n4931\n4939\n4948\n4973\n4974\n4980\n4984\n4994\n4996\n5000\n5008\n5043\n5045\n5047\n5065\n5071\n5077\n5090\n5092\n5093\n5111\n5115\n5116\n5124\n5125\n5126\n5137\n5148\n5153\n5154\n5155\n5157\n5163\n5166\n5167\n5176\n5186\n5187\n5190\n5200\n5227\n5230\n5236\n5241\n5247\n5251\n5254\n5256\n5279\n5286\n5287\n5295\n5311\n5319\n5342\n5344\n5354\n5357\n5359\n5376\n5378\n5380\n5381\n5392\n5400\n5405\n5408\n5420\n5423\n5428\n5438\n5449\n5451\n5478\n5492\n5496\n5502\n5503\n5508\n5511\n5523\n5527\n5528\n5529\n5532\n5537\n5543\n5547\n5556\n5567\n5572\n5573\n5574\n5579\n5580\n5610\n5618\n5640\n5641\n5644\n5647\n5689\n5690\n5691\n5695\n5697\n5700\n5720\n5723\n5724\n5727\n5729\n5751\n5754\n5757\n5760\n5778\n5789\n5792\n5797\n5801\n5806\n5808\n5819\n5828\n5830\n5866\n5868\n5871\n5874\n5875\n5879\n5880\n5891\n5897\n5908\n5924\n5925\n5927\n5947\n5968\n5973\n5987\n5988\n5994\n5995\n5999\n6001\n6016\n6024\n6033\n6041\n6042\n6045\n6048\n6062\n6069\n6070\n6077\n6094\n6119\n6127\n6128\n6130\n6145\n6147\n6155\n6159\n6160\n6170\n6179\n6183\n6194\n6201\n6204\n6205\n6207\n6210\n6211\n6213\n6218\n6237\n6259\n6263\n6265\n6267\n6272\n6279\n6289\n6291\n6298\n6310\n6311\n6354\n6360\n6364\n6365\n6371\n6373\n6386\n6400\n6406\n6413\n6419\n6432\n6433\n6442\n6443\n6444\n6466\n6469\n6478\n6488\n6489\n6501\n6512\n6513\n6522\n6530\n6533\n6534\n6537\n6538\n6547\n6554\n6555\n6575\n6578\n6582\n6584\n6590\n6599\n6610\n6611\n6629\n6638\n6647\n6649\n6672\n6684\n6692\n6707\n6708\n6720\n6727\n6738\n6742\n6744\n6754\n6755\n6759\n6762\n6774\n6780\n6787\n6793\n6803\n6815\n6822\n6823\n6827\n6828\n6848\n6852\n6862\n6877\n6885\n6888\n6908\n6919\n6929\n6947\n6966\n6970\n6972\n6983\n6993\n6999\n7013\n7017\n7022\n7024\n7028\n7033\n7042\n7055\n7063\n7064\n7067\n7074\n7077\n7084\n7089\n7093\n7112\n7114\n7116\n7121\n7125\n7128\n7134\n7137\n7145\n7148\n7154\n7160\n7162\n7173\n7188\n7194\n7210\n7211\n7220\n7222\n7224\n7227\n7242\n7250\n7263\n7264\n7270\n7293\n7298\n7301\n7306\n7318\n7320\n7324\n7331\n7334\n7338\n7358\n7361\n7373\n7390\n7397\n7398\n7411\n7420\n7424\n7429\n7439\n7441\n7443\n7444\n7445\n7455\n7484\n7500\n7502\n7506\n7511\n7512\n7518\n7539\n7554\n7566\n7574\n7578\n7582\n7596\n7597\n7605\n7609\n7613\n7618\n7625\n7654\n7655\n7660\n7664\n7665\n7666\n7672\n7674\n7681\n7689\n7693\n7708\n7723\n7727\n7766\n7774\n7796\n7806\n7808\n7814\n7815\n7823\n7830\n7831\n7835\n7841\n7842\n7843\n7848\n7850\n7854\n7857\n7859\n7867\n7872\n7877\n7883\n7906\n7911\n7927\n7930\n7931\n7940\n7943\n7948\n7950\n7956\n7966\n7971\n7972\n7977\n7978\n7979\n7987\n7994\n8001\n8003\n8011\n8032\n8044\n8045\n8052\n8068\n8072\n8080\n8090\n8102\n8103\n8115\n8137\n8140\n8141\n8143\n8149\n8162\n8164\n8165\n8181\n8183\n8185\n8206\n8211\n8225\n8249\n8252\n8268\n8273\n8277\n8282\n8290\n8304\n8311\n8315\n8325\n8338\n8342\n8350\n8351\n8368\n8369\n8370\n8371\n8373\n8382\n8384\n8386\n8387\n8389\n8398\n8415\n8416\n8434\n8440\n8444\n8446\n8453\n8457\n8485\n8488\n8489\n8491\n8497\n8504\n8520\n8522\n8523\n8531\n8541\n8543\n8545\n8551\n8570\n8574\n8575\n8577\n8584\n8587\n8590\n8604\n8610\n8624\n8629\n8631\n8632\n8638\n8640\n8647\n8651\n8668\n8679\n8686\n8694\n8705\n8715\n8720\n8722\n8723\n8725\n8730\n8734\n8738\n8745\n8748\n8752\n8760\n8767\n8776\n8784\n8786\n8789\n8800\n8801\n8809\n8820\n8845\n8847\n8853\n8856\n8865\n8870\n8871\n8890\n8891\n8895\n8899\n8911\n8916\n8917\n8931\n8944\n8953\n8959\n8963\n8970\n8982\n8992\n9000\n9002\n9003\n9015\n9017\n9020\n9021\n9023\n9032\n9034\n9040\n9044\n9048\n9072\n9074\n9083\n9095\n9096\n9108\n9113\n9120\n9121\n9134\n9144\n9145\n9150\n9151\n9163\n9172\n9174\n9178\n9179\n9184\n9204\n9216\n9224\n9227\n9233\n9237\n9240\n9256\n9257\n9260\n9261\n9264\n9265\n9286\n9289\n9291\n9299\n9308\n9314\n9326\n9331\n9332\n9336\n9337\n9339\n9341\n9343\n9345\n9355\n9358\n9397\n9414\n9417\n9419\n9423\n9437\n9438\n9443\n9450\n9455\n9460\n9462\n9464\n9469\n9472\n9486\n9496\n9499\n9520\n9522\n9527\n9529\n9530\n9537\n9539\n9544\n9550\n9556\n9559\n9560\n9576\n9577\n9585\n9598\n9600\n9612\n9615\n9617\n9619\n9634\n9637\n9647\n9654\n9668\n9677\n9686\n9691\n9701\n9728\n9750\n9755\n9758\n9768\n9787\n9792\n9799\n9802\n9803\n9807\n9808\n9811\n9812\n9814\n9815\n9820\n9825\n9829\n9831\n9847\n9848\n9860\n9874\n9876\n9879\n9885\n9895\n9901\n9919\n9921\n9937\n9943\n9947\n9955\n9971\n9972\n9985\n9986\n9994\n9997\n10000\n10006\n10007\n10022\n10028\n10046\n10061\n10062\n10069\n10090\n10091\n10094\n10111\n10113\n10129\n10142\n10146\n10161\n10170\n10173\n10174\n10180\n10181\n10200\n10212\n10219\n10222\n10226\n10227\n10228\n10230\n10232\n10235\n'
PINS_6 = '13\n14\n35\n38\n43\n78\n101\n112\n115\n136\n167\n175\n199\n200\n222\n224\n238\n267\n272\n283\n302\n303\n305\n318\n353\n380\n396\n402\n411\n438\n451\n473\n490\n501\n510\n515\n523\n532\n555\n561\n572\n584\n596\n621\n632\n633\n665\n670\n679\n700\n705\n729\n781\n784\n787\n793\n794\n804\n805\n812\n874\n885\n887\n923\n927\n936\n945\n956\n965\n967\n979\n993\n994\n998\n1001\n1004\n1038\n1039\n1044\n1075\n1076\n1078\n1082\n1091\n1112\n1118\n1122\n1126\n1166\n1173\n1178\n1202\n1216\n1219\n1224\n1246\n1277\n1282\n1320\n1323\n1341\n1343\n1345\n1361\n1370\n1390\n1395\n1405\n1431\n1438\n1445\n1454\n1468\n1489\n1498\n1515\n1516\n1530\n1537\n1550\n1585\n1595\n1601\n1615\n1618\n1651\n1652\n1664\n1674\n1679\n1686\n1693\n1703\n1718\n1732\n1762\n1763\n1786\n1806\n1815\n1824\n1827\n1828\n1868\n1874\n1891\n1909\n1938\n1961\n1963\n1968\n1973\n1983\n1991\n2009\n2021\n2024\n2052\n2053\n2081\n2082\n2083\n2085\n2091\n2104\n2105\n2109\n2111\n2114\n2115\n2152\n2155\n2158\n2164\n2207\n2231\n2268\n2270\n2272\n2285\n2287\n2306\n2308\n2320\n2333\n2348\n2351\n2363\n2379\n2420\n2430\n2456\n2465\n2471\n2475\n2484\n2495\n2501\n2502\n2507\n2546\n2568\n2569\n2575\n2591\n2617\n2618\n2625\n2628\n2648\n2651\n2679\n2682\n2706\n2717\n2725\n2729\n2734\n2745\n2756\n2758\n2794\n2803\n2807\n2808\n2819\n2828\n2853\n2855\n2858\n2875\n2880\n2901\n2916\n2923\n2927\n2929\n2944\n2961\n2973\n2976\n2990\n2997\n2998\n3005\n3022\n3055\n3071\n3088\n3111\n3117\n3122\n3129\n3132\n3133\n3141\n3165\n3195\n3200\n3217\n3232\n3238\n3265\n3266\n3271\n3280\n3292\n3299\n3331\n3341\n3357\n3361\n3371\n3372\n3375\n3394\n3397\n3417\n3430\n3440\n3461\n3462\n3477\n3482\n3503\n3510\n3518\n3557\n3575\n3577\n3612\n3620\n3635\n3636\n3648\n3655\n3666\n3670\n3682\n3683\n3690\n3709\n3715\n3721\n3731\n3744\n3751\n3756\n3773\n3776\n3790\n3805\n3836\n3844\n3847\n3894\n3898\n3900\n3911\n3945\n3950\n3955\n3988\n3991\n3992\n4002\n4003\n4005\n4026\n4042\n4044\n4049\n4053\n4056\n4069\n4073\n4076\n4079\n4082\n4090\n4101\n4122\n4124\n4128\n4138\n4142\n4147\n4148\n4149\n4155\n4170\n4178\n4191\n4226\n4230\n4252\n4266\n4277\n4286\n4301\n4339\n4348\n4357\n4367\n4369\n4372\n4383\n4387\n4412\n4416\n4434\n4439\n4462\n4467\n4494\n4506\n4510\n4541\n4542\n4546\n4550\n4583\n4588\n4592\n4593\n4594\n4600\n4602\n4609\n4613\n4621\n4636\n4655\n4675\n4724\n4734\n4736\n4742\n4763\n4776\n4778\n4785\n4786\n4800\n4805\n4809\n4811\n4834\n4839\n4855\n4858\n4868\n4871\n4878\n4884\n4894\n4899\n4906\n4907\n4912\n4915\n4922\n4929\n4931\n4939\n4973\n4974\n4980\n4984\n4996\n5000\n5047\n5065\n5071\n5077\n5090\n5092\n5093\n5116\n5125\n5153\n5154\n5155\n5176\n5186\n5187\n5190\n5200\n5227\n5236\n5251\n5256\n5279\n5286\n5311\n5342\n5344\n5354\n5357\n5359\n5376\n5378\n5392\n5400\n5420\n5423\n5449\n5451\n5478\n5492\n5496\n5503\n5523\n5527\n5528\n5529\n5532\n5543\n5547\n5567\n5573\n5574\n5579\n5610\n5618\n5640\n5641\n5689\n5690\n5691\n5695\n5720\n5723\n5751\n5754\n5778\n5789\n5792\n5797\n5801\n5806\n5808\n5819\n5830\n5866\n5868\n5875\n5879\n5891\n5924\n5925\n5927\n5947\n5968\n5973\n5988\n5994\n5995\n5999\n6016\n6045\n6048\n6069\n6070\n6077\n6094\n6119\n6127\n6130\n6145\n6159\n6183\n6194\n6201\n6205\n6213\n6237\n6267\n6289\n6291\n6364\n6373\n6400\n6413\n6432\n6433\n6444\n6466\n6469\n6478\n6488\n6489\n6501\n6512\n6513\n6530\n6533\n6534\n6538\n6547\n6554\n6575\n6582\n6590\n6611\n6629\n6647\n6649\n6692\n6707\n6708\n6720\n6727\n6738\n6742\n6744\n6754\n6762\n6774\n6787\n6793\n6803\n6815\n6823\n6827\n6828\n6848\n6852\n6877\n6885\n6908\n6929\n6947\n6966\n6970\n6972\n6983\n7017\n7022\n7055\n7067\n7074\n7077\n7084\n7089\n7093\n7114\n7116\n7121\n7128\n7145\n7148\n7154\n7162\n7173\n7188\n7194\n7211\n7220\n7222\n7224\n7227\n7242\n7263\n7264\n7270\n7298\n7301\n7318\n7320\n7324\n7331\n7338\n7358\n7361\n7373\n7397\n7411\n7420\n7424\n7429\n7439\n7443\n7444\n7445\n7455\n7484\n7500\n7506\n7511\n7512\n7518\n7539\n7566\n7574\n7578\n7582\n7596\n7609\n7613\n7618\n7654\n7655\n7660\n7665\n7666\n7672\n7674\n7681\n7689\n7693\n7723\n7727\n7766\n7774\n7806\n7814\n7815\n7831\n7835\n7842\n7843\n7848\n7850\n7857\n7859\n7872\n7877\n7883\n7906\n7911\n7927\n7930\n7940\n7943\n7948\n7950\n7956\n7966\n7971\n7972\n7978\n7987\n8001\n8003\n8011\n8032\n8044\n8045\n8052\n8072\n8103\n8115\n8137\n8141\n8143\n8149\n8162\n8164\n8165\n8211\n8225\n8249\n8268\n8273\n8282\n8290\n8304\n8311\n8315\n8325\n8338\n8342\n8350\n8369\n8370\n8371\n8387\n8389\n8398\n8415\n8416\n8434\n8440\n8444\n8446\n8457\n8485\n8488\n8489\n8497\n8504\n8520\n8522\n8523\n8531\n8541\n8545\n8551\n8570\n8574\n8575\n8584\n8587\n8590\n8610\n8638\n8640\n8647\n8686\n8705\n8720\n8725\n8730\n8745\n8748\n8752\n8767\n8786\n8789\n8800\n8801\n8809\n8820\n8845\n8853\n8865\n8870\n8871\n8890\n8895\n8899\n8911\n8916\n8917\n8944\n8953\n8959\n8963\n8970\n8982\n8992\n9002\n9003\n9017\n9020\n9032\n9040\n9044\n9048\n9072\n9074\n9121\n9134\n9144\n9145\n9163\n9172\n9174\n9178\n9179\n9184\n9204\n9227\n9237\n9240\n9257\n9261\n9286\n9289\n9291\n9299\n9308\n9314\n9332\n9337\n9339\n9341\n9345\n9397\n9414\n9419\n9437\n9443\n9450\n9455\n9464\n9472\n9486\n9520\n9527\n9529\n9530\n9537\n9544\n9550\n9559\n9576\n9598\n9600\n9612\n9615\n9617\n9619\n9634\n9637\n9647\n9654\n9668\n9686\n9701\n9728\n9755\n9768\n9792\n9807\n9811\n9812\n9814\n9820\n9825\n9831\n9848\n9876\n9879\n9901\n9919\n9921\n9937\n9955\n9972\n9986\n9994\n10000\n10007\n10022\n10028\n10046\n10061\n10062\n10090\n10091\n10094\n10142\n10146\n10170\n10173\n10174\n10181\n10200\n10212\n10219\n10222\n10226\n10227\n'

# JOIN4 executor C (staged + pins + section timers), injected at build.
JOIN4_C = '// JOIN4 integrated executor: staged bounded expert cache + static pins +\n// section timers + decode/prefill split + TTFT.\n//\n// Base: staged-q2k BOUNDED_C (v10-style async pread-into-slots, proven on\n// this pin) + ROUTE_C. Delta vs staged (ONLY these):\n//   - GGML_PHASE6_SLOTS=N: exact slot count (preferred over CACHE_BYTES)\n//   - GGML_PHASE6_PINS=/path: static pins, one (layer<<8)|expert int/line;\n//     pinned bundles never evicted; PRELOADED at init (pins resident t=0,\n//     matching sim semantics); preload bytes/nanoseconds counted separately\n//   - GGML_PHASE6_PROFILE=1: barrier-flushed per-node section timers\n//     (thread 0; exact by construction: every node lands in one bucket):\n//     sections ATTN/GDN/MOE/SHARED/LMHEAD/MISC via cb-name markers,\n//     EXPERT = MUL_MAT_ID on _exps (fetch_prep subtracted for compute),\n//     ROUTER_K2 = MOE section minus expert minus fetch\n//   - decode/prefill split (MoE ids rows==1 -> decode), decode token count,\n//     TTFT (first prefill start -> first decode graph end)\n// Works with bounded ON or OFF (resident arm: same binary, env unset;\n// timers independent of the cache).\n//\n// Injected into ggml/src/ggml-cpu/ggml-cpu.c at the g_state anchor (same\n// pin 3057bb6 as staged). Timer hooks called from the graph thread loop\n// (join4_node_start at node top, join4_graph_end at loop end, ith==0).\n// K1K2 graph patch (llama-graph.cpp) + iqp redirect + loader/laizy-model\n// patches are separate anchors (see kernel); this file is ggml-cpu.c only.\n\n// Phase 7B pipeline explicit byte-bounded expert store.  The loader registers the\n// original GGUF offsets, while this C backend owns only fixed anonymous slots.\n#if defined(__linux__)\n#include <pthread.h>\n#include <sched.h>\n#include <sys/mman.h>\n#include <sys/stat.h>\n#include <unistd.h>\n#endif\n\nstruct phase6_record {\n    const struct ggml_tensor * tensor;\n    int fd;\n    size_t base_offset;\n    size_t slice_bytes;\n    int layer;\n    int kind; // gate=0, up=1, down=2\n    int valid;\n};\n\nstruct phase6_slot {\n    int layer;\n    int expert;\n    uint64_t age;\n    int valid;\n    int ready_mask;\n};\n\nstruct phase6_async_task {\n    int layer;\n    int expert;\n    int slot;\n    size_t read_bytes;\n    uint64_t read_ns;\n    int decode;\n};\n\nstatic struct phase6_record phase6_records[40][3];\nstatic int phase6_bundle_slots[40][256];\nstatic struct phase6_slot * phase6_slots = NULL;\nstatic unsigned char * phase6_storage = NULL;\nstatic size_t phase6_slot_bytes = 0;\nstatic size_t phase6_slot_count = 0;\nstatic uint64_t phase6_age = 0;\nstatic uint64_t phase6_requests = 0;\nstatic uint64_t phase6_hits = 0;\nstatic uint64_t phase6_misses = 0;\nstatic uint64_t phase6_evictions = 0;\nstatic uint64_t phase6_read_calls = 0;\nstatic uint64_t phase6_read_bytes = 0;\nstatic uint64_t phase6_read_ns = 0;\nstatic uint64_t phase6_async_tasks = 0;\nstatic uint64_t phase6_async_wait_ns = 0;\nstatic uint64_t phase6_ready_wait_ns = 0;\nstatic uint64_t phase6_ready_wait_events = 0;\nstatic int phase6_initialized = 0;\nstatic int phase6_report_registered = 0;\n// JOIN4: decode-only request counters + mode flag (ids rows==1 -> decode)\nstatic uint64_t phase6_dec_requests = 0;\nstatic uint64_t phase6_dec_hits = 0;\nstatic uint64_t phase6_dec_misses = 0;\nstatic uint64_t phase6_dec_read_bytes = 0;\nstatic uint64_t phase6_dec_read_ns = 0;\nstatic int phase6_decode_mode = 0;\n// JOIN4: static pins (bitset over (layer<<8)|expert, universe 10240)\nenum { PHASE6_PIN_WORDS = 160 };\nstatic uint64_t phase6_pin_bits[PHASE6_PIN_WORDS];\nstatic uint64_t phase6_pins_loaded = 0;\nstatic uint64_t phase6_preload_bytes = 0;\nstatic uint64_t phase6_preload_ns = 0;\nstatic uint64_t phase6_pin_violations = 0;\nenum { PHASE6_MAX_ASYNC_TASKS = 4096 };\nstatic struct phase6_async_task phase6_async_task_records[PHASE6_MAX_ASYNC_TASKS];\nstatic pthread_t phase6_async_threads[PHASE6_MAX_ASYNC_TASKS];\nstatic int phase6_async_task_count = 0;\nstatic int phase6_async_started_count = 0;\nstatic int phase6_async_layer = -1;\nstatic void phase6_reap_async(void);\n\nstatic int phase6_enabled(void) {\n    const char * value = getenv("GGML_PHASE6_BOUNDED_CACHE");\n    return value != NULL && atoi(value) != 0;\n}\n\nstatic int phase6_async_enabled(void) {\n    const char * value = getenv("GGML_PHASE6_ASYNC");\n    return value != NULL && atoi(value) != 0;\n}\n\nstatic int join4_profile_enabled(void) {\n    const char * value = getenv("GGML_PHASE6_PROFILE");\n    return value != NULL && atoi(value) != 0;\n}\n\n// Cached flag for the per-node dispatch hook (avoids getenv per node).\nstatic int join4_prof_cached = -1;\nstatic int join4_prof_on(void) {\n    if (join4_prof_cached < 0)\n        join4_prof_cached = join4_profile_enabled() ? 1 : 0;\n    return join4_prof_cached;\n}\n\n// Zero-copy challenger state: kernels consume expert bytes directly from a\n// file-backed MAP_SHARED mapping of the GGUF.  Slots stay purely logical\n// (same LRU policy/count as the control); eviction bounds RSS with\n// MADV_DONTNEED on the evicted bundle\'s file ranges.  No pread, no slot\n// copies, no reader threads.\n// JOIN4: vehicle dead (phase10i verdict); code kept verbatim, never enabled.\nstatic unsigned char * phase6_zc_file_map = NULL;\nstatic size_t phase6_zc_file_size = 0;\nstatic uint64_t phase6_zc_madvise_ns = 0;\nstatic uint64_t phase6_zc_madvise_calls = 0;\nstatic uint64_t phase6_zc_madvise_bytes = 0;\nstatic uint64_t phase6_zc_madvise_errors = 0;\nenum { PHASE6_ZC_EVICT_RING = 64 };\nstatic int phase6_zc_evict_layer[PHASE6_ZC_EVICT_RING];\nstatic int phase6_zc_evict_expert[PHASE6_ZC_EVICT_RING];\nstatic int phase6_zc_evict_count = 0;\n\nstatic int phase6_zc_enabled(void) {\n    const char * value = getenv("GGML_PHASE6_ZERO_COPY");\n    return phase6_enabled() && value != NULL && atoi(value) != 0;\n}\n\nstatic uint64_t phase6_now_ns(void) {\n    struct timespec ts;\n    clock_gettime(CLOCK_MONOTONIC, &ts);\n    return (uint64_t) ts.tv_sec * 1000000000ULL + (uint64_t) ts.tv_nsec;\n}\n\n// JOIN4 section timers: barrier-flushed per-node attribution (thread 0).\n// Every node lands in exactly one bucket => sum == graph wall by design.\nenum {\n    JOIN4_SEC_MISC = 0, JOIN4_SEC_ATTN_TENT, JOIN4_SEC_MOE, JOIN4_SEC_SHARED,\n    JOIN4_SEC_LMHEAD, JOIN4_NSECS\n};\nstatic uint64_t join4_dec_ns[JOIN4_NSECS];\nstatic uint64_t join4_pre_ns[JOIN4_NSECS];\nstatic uint64_t join4_dec_cnt[JOIN4_NSECS];\nstatic uint64_t join4_pre_cnt[JOIN4_NSECS];\nstatic uint64_t join4_dec_attn_ns = 0;   // full-attention section\nstatic uint64_t join4_pre_attn_ns = 0;\nstatic uint64_t join4_dec_gdn_ns = 0;    // gated-delta-net section\nstatic uint64_t join4_pre_gdn_ns = 0;\nstatic uint64_t join4_scratch_dec_ns = 0; // ATTN_TENT accumulates here...\nstatic uint64_t join4_scratch_pre_ns = 0; // ...resolved to ATTN/GDN at close\nstatic uint64_t join4_dec_expert_ns = 0; // MUL_MAT_ID on _exps (node wall)\nstatic uint64_t join4_pre_expert_ns = 0;\nstatic uint64_t join4_dec_expert_cnt = 0;\nstatic uint64_t join4_pre_expert_cnt = 0;\nstatic uint64_t join4_dec_fetchprep_ns = 0; // in-node fetch (subtracted)\nstatic uint64_t join4_pre_fetchprep_ns = 0;\nstatic int join4_section = JOIN4_SEC_MISC;\nstatic int join4_decode_mode = 0; // init PREFILL: first graph is prefill\nstatic int join4_pending_valid = 0;\nstatic uint64_t join4_pending_t0 = 0;\nstatic int join4_pending_bucket = JOIN4_SEC_MISC;\nstatic int join4_pending_expert = 0;\nstatic int join4_pending_resolve = 0; // 0=none, 1=attn, 2=gdn (closer nodes)\nstatic int join4_pending_wb = -1; // weight bucket, -1 = not a matmul\nstatic uint64_t join4_markers_seen = 0;\nstatic uint64_t join4_force_closes = 0;\n// JOIN4b: weight-bucket timers. EVERY MUL_MAT/MUL_MAT_ID attributed by its\n// src0 (weight) tensor name from the loader: immutable, always present,\n// independent of cb-name sections (v2 sections leaked unnamed bodies into\n// MISC: shexp matmuls run before their "ffn_shexp" opener, etc.).\n// Both attributions run (sections + weight buckets); residual cross-checks.\nenum {\n    JOIN4_WB_EXPS = 0, JOIN4_WB_ATTN, JOIN4_WB_GDN, JOIN4_WB_SHEXP,\n    JOIN4_WB_ROUTER, JOIN4_WB_OUT, JOIN4_WB_OTHER, JOIN4_WB_N\n};\nstatic uint64_t join4_dec_wb[JOIN4_WB_N];\nstatic uint64_t join4_pre_wb[JOIN4_WB_N];\nstatic uint64_t join4_dec_wb_cnt[JOIN4_WB_N];\nstatic uint64_t join4_pre_wb_cnt[JOIN4_WB_N];\nstatic uint64_t join4_wb_unmatched_logged = 0;\n\nstatic int join4_wbucket(const char * wname) {\n    if (wname == NULL) return JOIN4_WB_OTHER;\n    if (strstr(wname, "_exps")) return JOIN4_WB_EXPS;\n    if (strstr(wname, "attn_q") || strstr(wname, "attn_k") ||\n        strstr(wname, "attn_v") || strstr(wname, "attn_o") ||\n        strstr(wname, "qkv")) return JOIN4_WB_ATTN;\n    if (strstr(wname, "ssm_") || strstr(wname, "conv") ||\n        strstr(wname, "delta")) return JOIN4_WB_GDN;\n    if (strstr(wname, "shexp")) return JOIN4_WB_SHEXP;\n    if (strstr(wname, "ffn_gate_inp")) return JOIN4_WB_ROUTER;\n    if (strstr(wname, "output")) return JOIN4_WB_OUT;\n    return JOIN4_WB_OTHER;\n}\n\n// Nodelist ground truth: first graph\'s node sequence (env-gated, 1 run).\nstatic FILE * join4_nodelist_fp = NULL;\nstatic int join4_nodelist_done = 0;\nstatic uint64_t join4_decode_graphs = 0;\nstatic uint64_t join4_prefill_graphs = 0;\nstatic uint64_t join4_graph_moe_nodes = 0;\nstatic uint64_t join4_prefill_start_ns = 0;\nstatic uint64_t join4_started = 0;\nstatic uint64_t join4_ttft_ns = 0;\n\nstatic void phase6_zc_evict_range(int layer, int expert) {\n    if (phase6_zc_file_map == NULL) return;\n    const uint64_t start = phase6_now_ns();\n    long page = 4096;\n#if defined(__linux__)\n    const long configured = sysconf(_SC_PAGESIZE);\n    if (configured > 0) page = configured;\n#endif\n    const size_t psize = (size_t) page;\n    for (int kind = 0; kind < 3; ++kind) {\n        const struct phase6_record * record = &phase6_records[layer][kind];\n        const size_t begin = record->base_offset + (size_t) expert * record->slice_bytes;\n        const size_t end = begin + record->slice_bytes;\n        const size_t aligned_begin = (begin / psize) * psize;\n        const size_t aligned_end = ((end + psize - 1) / psize) * psize;\n        if (aligned_end > phase6_zc_file_size) continue;\n#if defined(__linux__)\n        if (madvise(phase6_zc_file_map + aligned_begin, aligned_end - aligned_begin, MADV_DONTNEED) != 0) {\n            phase6_zc_madvise_errors++;\n        }\n#endif\n        phase6_zc_madvise_bytes += aligned_end - aligned_begin;\n    }\n    phase6_zc_madvise_calls += 3;\n    phase6_zc_madvise_ns += phase6_now_ns() - start;\n}\n\nstatic int phase6_kind(const char * name) {\n    if (name == NULL || strstr(name, "ffn_") == NULL || strstr(name, "_exps") == NULL) return -1;\n    if (strstr(name, "ffn_gate_exps")) return 0;\n    if (strstr(name, "ffn_up_exps")) return 1;\n    if (strstr(name, "ffn_down_exps")) return 2;\n    return -1;\n}\n\nstatic int phase6_layer(const char * name) {\n    int layer = -1;\n    if (name != NULL) sscanf(name, "blk.%d.", &layer);\n    return layer;\n}\n\nvoid ggml_cpu_phase6_register_lazy_tensor(const struct ggml_tensor * tensor,\n        int fd, size_t base_offset, size_t nbytes, size_t expert_stride) {\n    if (!phase6_enabled() || tensor == NULL || tensor->name == NULL) return;\n    const int layer = phase6_layer(tensor->name);\n    const int kind = phase6_kind(tensor->name);\n    if (layer < 0 || layer >= 40 || kind < 0 || kind >= 3 || expert_stride == 0 ||\n            expert_stride * 256 > nbytes) return;\n    // llama_model_loader owns the original descriptor and may destroy it\n    // after model construction.  The explicit cache owns this duplicate so\n    // pread remains valid during generation.\n    const int owned_fd = dup(fd);\n    if (owned_fd < 0) {\n        fprintf(stderr, "PHASE6_BOUNDED_CACHE_ERROR dup failed errno=%d\\n", errno);\n        abort();\n    }\n    phase6_records[layer][kind] = (struct phase6_record) {\n        tensor, owned_fd, base_offset, expert_stride, layer, kind, 1\n    };\n}\n\nstatic void phase6_fail(const char * message) {\n    fprintf(stderr, "PHASE6_BOUNDED_CACHE_ERROR %s\\n", message);\n    abort();\n}\n\n// JOIN4: load static pins (one (layer<<8)|expert int per line).\nstatic void phase6_load_pins(void) {\n    const char * path = getenv("GGML_PHASE6_PINS");\n    if (path == NULL || path[0] == \'\\0\') return;\n    FILE * fp = fopen(path, "r");\n    if (fp == NULL) phase6_fail("pins file not readable");\n    char line[256];\n    while (fgets(line, sizeof(line), fp) != NULL) {\n        char * end = NULL;\n        const long key = strtol(line, &end, 10);\n        if (end == line || key < 0 || key >= 10240) phase6_fail("bad pin key");\n        phase6_pin_bits[((uint32_t) key >> 6) % PHASE6_PIN_WORDS] |= 1ULL << (key & 63);\n        phase6_pins_loaded++;\n    }\n    fclose(fp);\n}\n\nstatic int phase6_pinned(int layer, int expert) {\n    const uint32_t key = ((uint32_t) layer << 8) | (uint32_t) expert;\n    return (int) ((phase6_pin_bits[(key >> 6) % PHASE6_PIN_WORDS] >> (key & 63)) & 1ULL);\n}\n\nstatic void phase6_copy_read(int fd, void * dst, size_t bytes, size_t offset) {\n    size_t done = 0;\n    while (done < bytes) {\n        const ssize_t got = pread(fd, (char *) dst + done, bytes - done, offset + done);\n        if (got < 0) {\n            if (errno == EINTR) continue;\n            phase6_fail("async pread failed");\n        }\n        if (got == 0) phase6_fail("async short pread");\n        done += (size_t) got;\n    }\n}\n\nstatic void phase6_report(void) {\n    if (!phase6_enabled() && !join4_profile_enabled()) return;\n    if (phase6_enabled() && phase6_async_enabled()) phase6_reap_async();\n    if (phase6_enabled()) {\n    fprintf(stderr,\n        "PHASE6_BOUNDED_CACHE slot_bytes=%zu slots=%zu reserved_bytes=%zu "\n        "requests=%llu hits=%llu misses=%llu evictions=%llu read_calls=%llu "\n        "read_bytes=%llu read_ns=%llu async_tasks=%llu async_wait_ns=%llu "\n        "ready_wait_ns=%llu ready_wait_events=%llu "\n        "pins=%llu preload_bytes=%llu preload_ns=%llu pin_violations=%llu "\n        "dec_requests=%llu dec_hits=%llu dec_misses=%llu "\n        "dec_read_bytes=%llu dec_read_ns=%llu\\n",\n        phase6_slot_bytes, phase6_slot_count, phase6_slot_bytes * phase6_slot_count,\n        (unsigned long long) phase6_requests, (unsigned long long) phase6_hits,\n        (unsigned long long) phase6_misses, (unsigned long long) phase6_evictions,\n        (unsigned long long) phase6_read_calls, (unsigned long long) phase6_read_bytes,\n        (unsigned long long) phase6_read_ns, (unsigned long long) phase6_async_tasks,\n        (unsigned long long) phase6_async_wait_ns,\n        (unsigned long long) phase6_ready_wait_ns,\n        (unsigned long long) phase6_ready_wait_events,\n        (unsigned long long) phase6_pins_loaded,\n        (unsigned long long) phase6_preload_bytes,\n        (unsigned long long) phase6_preload_ns,\n        (unsigned long long) phase6_pin_violations,\n        (unsigned long long) phase6_dec_requests,\n        (unsigned long long) phase6_dec_hits,\n        (unsigned long long) phase6_dec_misses,\n        (unsigned long long) phase6_dec_read_bytes,\n        (unsigned long long) phase6_dec_read_ns);\n    }\n    if (join4_profile_enabled()) {\n    fprintf(stderr,\n        "PHASE6_PROFILE dec_graphs=%llu pre_graphs=%llu ttft_ns=%llu "\n        "dec_attn_ns=%llu dec_gdn_ns=%llu dec_moe_rest_ns=%llu "\n        "dec_expert_node_ns=%llu dec_expert_cnt=%llu dec_fetchprep_ns=%llu "\n        "dec_shared_ns=%llu dec_lmhead_ns=%llu dec_misc_ns=%llu "\n        "pre_attn_ns=%llu pre_gdn_ns=%llu pre_moe_rest_ns=%llu "\n        "pre_expert_node_ns=%llu pre_expert_cnt=%llu pre_fetchprep_ns=%llu "\n        "pre_shared_ns=%llu pre_lmhead_ns=%llu pre_misc_ns=%llu "\n        "markers=%llu force_closes=%llu "\n        "wb_dec_exps=%llu wb_dec_attn=%llu wb_dec_gdn=%llu "\n        "wb_dec_shexp=%llu wb_dec_router=%llu wb_dec_out=%llu "\n        "wb_dec_other=%llu wb_pre_exps=%llu wb_pre_attn=%llu "\n        "wb_pre_gdn=%llu wb_pre_shexp=%llu wb_pre_router=%llu "\n        "wb_pre_out=%llu wb_pre_other=%llu\\n",\n        (unsigned long long) join4_decode_graphs,\n        (unsigned long long) join4_prefill_graphs,\n        (unsigned long long) join4_ttft_ns,\n        (unsigned long long) join4_dec_attn_ns,\n        (unsigned long long) join4_dec_gdn_ns,\n        (unsigned long long) join4_dec_ns[JOIN4_SEC_MOE],\n        (unsigned long long) join4_dec_expert_ns,\n        (unsigned long long) join4_dec_expert_cnt,\n        (unsigned long long) join4_dec_fetchprep_ns,\n        (unsigned long long) join4_dec_ns[JOIN4_SEC_SHARED],\n        (unsigned long long) join4_dec_ns[JOIN4_SEC_LMHEAD],\n        (unsigned long long) join4_dec_ns[JOIN4_SEC_MISC],\n        (unsigned long long) join4_pre_attn_ns,\n        (unsigned long long) join4_pre_gdn_ns,\n        (unsigned long long) join4_pre_ns[JOIN4_SEC_MOE],\n        (unsigned long long) join4_pre_expert_ns,\n        (unsigned long long) join4_pre_expert_cnt,\n        (unsigned long long) join4_pre_fetchprep_ns,\n        (unsigned long long) join4_pre_ns[JOIN4_SEC_SHARED],\n        (unsigned long long) join4_pre_ns[JOIN4_SEC_LMHEAD],\n        (unsigned long long) join4_pre_ns[JOIN4_SEC_MISC],\n        (unsigned long long) join4_markers_seen,\n        (unsigned long long) join4_force_closes,\n        (unsigned long long) join4_dec_wb[JOIN4_WB_EXPS],\n        (unsigned long long) join4_dec_wb[JOIN4_WB_ATTN],\n        (unsigned long long) join4_dec_wb[JOIN4_WB_GDN],\n        (unsigned long long) join4_dec_wb[JOIN4_WB_SHEXP],\n        (unsigned long long) join4_dec_wb[JOIN4_WB_ROUTER],\n        (unsigned long long) join4_dec_wb[JOIN4_WB_OUT],\n        (unsigned long long) join4_dec_wb[JOIN4_WB_OTHER],\n        (unsigned long long) join4_pre_wb[JOIN4_WB_EXPS],\n        (unsigned long long) join4_pre_wb[JOIN4_WB_ATTN],\n        (unsigned long long) join4_pre_wb[JOIN4_WB_GDN],\n        (unsigned long long) join4_pre_wb[JOIN4_WB_SHEXP],\n        (unsigned long long) join4_pre_wb[JOIN4_WB_ROUTER],\n        (unsigned long long) join4_pre_wb[JOIN4_WB_OUT],\n        (unsigned long long) join4_pre_wb[JOIN4_WB_OTHER]);\n    }\n    if (phase6_enabled()) {\n    if (phase6_zc_enabled()) {\n        long admitted_pages = 0;\n        long admitted_resident = 0;\n        long evicted_pages = 0;\n        long evicted_resident = 0;\n#if defined(__linux__)\n        long page = 4096;\n        const long configured = sysconf(_SC_PAGESIZE);\n        if (configured > 0) page = configured;\n        const size_t psize = (size_t) page;\n        unsigned char vec = 0;\n        long sampled = 0;\n        for (size_t i = 0; i < phase6_slot_count && sampled < 96; ++i) {\n            if (!phase6_slots[i].valid) continue;\n            const int layer = phase6_slots[i].layer;\n            const int expert = phase6_slots[i].expert;\n            const struct phase6_record * record = &phase6_records[layer][0];\n            const size_t mid = record->base_offset + (size_t) expert * record->slice_bytes\n                + record->slice_bytes / 2;\n            const size_t aligned = (mid / psize) * psize;\n            if (aligned + psize > phase6_zc_file_size) continue;\n            vec = 0;\n            if (mincore(phase6_zc_file_map + aligned, psize, &vec) == 0) {\n                admitted_pages++;\n                if (vec & 1) admitted_resident++;\n            }\n            sampled++;\n        }\n        const int ring = phase6_zc_evict_count < PHASE6_ZC_EVICT_RING\n            ? phase6_zc_evict_count : PHASE6_ZC_EVICT_RING;\n        for (int i = 0; i < ring; ++i) {\n            const int layer = phase6_zc_evict_layer[i];\n            const int expert = phase6_zc_evict_expert[i];\n            if (layer < 0 || phase6_bundle_slots[layer][expert] >= 0) continue;\n            const struct phase6_record * record = &phase6_records[layer][0];\n            const size_t mid = record->base_offset + (size_t) expert * record->slice_bytes\n                + record->slice_bytes / 2;\n            const size_t aligned = (mid / psize) * psize;\n            if (aligned + psize > phase6_zc_file_size) continue;\n            vec = 0;\n            if (mincore(phase6_zc_file_map + aligned, psize, &vec) == 0) {\n                evicted_pages++;\n                if (vec & 1) evicted_resident++;\n            }\n        }\n#endif\n        long smaps_rss_kb = -1, smaps_anon_kb = -1, smaps_file_kb = -1;\n#if defined(__linux__)\n        FILE * smaps = fopen("/proc/self/smaps_rollup", "r");\n        if (smaps != NULL) {\n            char line[256];\n            while (fgets(line, sizeof(line), smaps) != NULL) {\n                if (strncmp(line, "Rss:", 4) == 0) smaps_rss_kb = atol(line + 4);\n                else if (strncmp(line, "Anonymous:", 10) == 0) smaps_anon_kb = atol(line + 10);\n                else if (strncmp(line, "FilePmdMapped:", 14) == 0) { /* skip */ }\n                else if (line[0] == \'F\' && strncmp(line, "FileRSS:", 8) == 0) { /* older kernels */ }\n            }\n            fclose(smaps);\n            // smaps_rollup reports Rss/Anonymous; derive file as Rss - Anonymous.\n            if (smaps_rss_kb >= 0 && smaps_anon_kb >= 0) smaps_file_kb = smaps_rss_kb - smaps_anon_kb;\n        }\n#endif\n        fprintf(stderr,\n            "PHASE6_ZERO_COPY file_bytes=%zu madvise_calls=%llu madvise_bytes=%llu "\n            "madvise_ns=%llu madvise_errors=%llu admitted_pages=%ld admitted_resident=%ld "\n            "evicted_pages=%ld evicted_resident=%ld smaps_rss_kb=%ld smaps_anon_kb=%ld smaps_file_kb=%ld\\n",\n            phase6_zc_file_size,\n            (unsigned long long) phase6_zc_madvise_calls,\n            (unsigned long long) phase6_zc_madvise_bytes,\n            (unsigned long long) phase6_zc_madvise_ns,\n            (unsigned long long) phase6_zc_madvise_errors,\n            admitted_pages, admitted_resident, evicted_pages, evicted_resident,\n            smaps_rss_kb, smaps_anon_kb, smaps_file_kb);\n    }\n#if defined(__linux__)\n    for (int layer = 0; layer < 40; ++layer) {\n        for (int kind = 0; kind < 3; ++kind) {\n            if (phase6_records[layer][kind].valid && phase6_records[layer][kind].fd >= 0) {\n                close(phase6_records[layer][kind].fd);\n                phase6_records[layer][kind].fd = -1;\n            }\n        }\n    }\n#endif\n    }\n}\n\nstatic void phase6_init(void) {\n    if (phase6_initialized) return;\n    for (int layer = 0; layer < 40; ++layer) {\n        for (int expert = 0; expert < 256; ++expert) phase6_bundle_slots[layer][expert] = -1;\n    }\n    for (int layer = 0; layer < 40; ++layer) {\n        size_t total = 0;\n        for (int kind = 0; kind < 3; ++kind) {\n            if (!phase6_records[layer][kind].valid) phase6_fail("incomplete routed tensor registration");\n            total += phase6_records[layer][kind].slice_bytes;\n        }\n        if (layer == 0) phase6_slot_bytes = total;\n        if (total != phase6_slot_bytes) phase6_fail("nonuniform bundle size is not supported by v1");\n    }\n    // JOIN4: exact slot count preferred; byte capacity is the fallback.\n    const char * slots_text = getenv("GGML_PHASE6_SLOTS");\n    if (slots_text != NULL && slots_text[0] != \'\\0\') {\n        phase6_slot_count = (size_t) strtoull(slots_text, NULL, 10);\n    } else {\n        const char * capacity_text = getenv("GGML_PHASE6_CACHE_BYTES");\n        const unsigned long long capacity = capacity_text ? strtoull(capacity_text, NULL, 10) : 0;\n        phase6_slot_count = phase6_slot_bytes ? (size_t) (capacity / phase6_slot_bytes) : 0;\n    }\n    if (phase6_slot_count == 0) phase6_fail("cache capacity does not hold one routed bundle");\n    phase6_load_pins();\n    if ((size_t) phase6_pins_loaded > phase6_slot_count) phase6_fail("more pins than slots");\n    phase6_slots = (struct phase6_slot *) calloc(phase6_slot_count, sizeof(*phase6_slots));\n    if (phase6_zc_enabled()) {\n        // Challenger: map the GGUF file-backed and consume expert bytes\n        // directly.  Slots stay logical only; no anonymous storage.\n        for (int i = 0; i < PHASE6_ZC_EVICT_RING; ++i) {\n            phase6_zc_evict_layer[i] = -1;\n            phase6_zc_evict_expert[i] = -1;\n        }\n        const int map_fd = phase6_records[0][0].fd;\n        struct stat st;\n        if (map_fd < 0 || fstat(map_fd, &st) != 0 || st.st_size <= 0) {\n            phase6_fail("zero-copy fstat failed");\n        }\n        phase6_zc_file_size = (size_t) st.st_size;\n#if defined(__linux__)\n        phase6_zc_file_map = (unsigned char *) mmap(NULL, phase6_zc_file_size,\n            PROT_READ, MAP_SHARED, map_fd, 0);\n        if (phase6_zc_file_map == MAP_FAILED) phase6_zc_file_map = NULL;\n        // v2 fix: MADV_RANDOM disables readahead AND fault-around.  v1 leaked\n        // file RSS (4.02 GiB, still climbing) via fault-around zombie ptes:\n        // the 64 KB speculative window re-mapped evicted pages adjacent to\n        // admitted slices without re-admission.\n        if (phase6_zc_file_map != NULL) {\n            madvise(phase6_zc_file_map, phase6_zc_file_size, MADV_RANDOM);\n        }\n#endif\n        if (phase6_zc_file_map == NULL || phase6_slots == NULL) {\n            phase6_fail("zero-copy file mapping failed");\n        }\n    } else {\n#if defined(__linux__)\n        phase6_storage = (unsigned char *) mmap(NULL, phase6_slot_bytes * phase6_slot_count,\n            PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);\n        if (phase6_storage == MAP_FAILED) phase6_storage = NULL;\n#endif\n        if (phase6_storage == NULL || phase6_slots == NULL) phase6_fail("fixed cache allocation failed");\n    }\n    // JOIN4: preload pins (sync; one-time; counted separately from traffic).\n    if (phase6_pins_loaded > 0 && !phase6_zc_enabled()) {\n        const uint64_t t0 = phase6_now_ns();\n        for (int layer = 0; layer < 40; ++layer) {\n            for (int expert = 0; expert < 256; ++expert) {\n                if (!phase6_pinned(layer, expert)) continue;\n                int slot = -1;\n                for (size_t i = 0; i < phase6_slot_count; ++i) {\n                    if (!phase6_slots[i].valid) { slot = (int) i; break; }\n                }\n                if (slot < 0) phase6_fail("pin preload found no free slot");\n                phase6_slots[slot] = (struct phase6_slot) {\n                    layer, expert, ++phase6_age, 1, 7\n                };\n                phase6_bundle_slots[layer][expert] = slot;\n                unsigned char * dst = phase6_storage + (size_t) slot * phase6_slot_bytes;\n                size_t in_slot = 0;\n                for (int kind = 0; kind < 3; ++kind) {\n                    const struct phase6_record * record = &phase6_records[layer][kind];\n                    phase6_copy_read(record->fd, dst + in_slot, record->slice_bytes,\n                        record->base_offset + (size_t) expert * record->slice_bytes);\n                    in_slot += record->slice_bytes;\n                    phase6_preload_bytes += record->slice_bytes;\n                }\n            }\n        }\n        phase6_preload_ns = phase6_now_ns() - t0;\n    }\n    if (!phase6_report_registered) {\n        phase6_report_registered = 1;\n        atexit(phase6_report);\n    }\n    phase6_initialized = 1;\n}\n\nstatic int phase6_find_slot(int layer, int expert) {\n    if (layer < 0 || layer >= 40 || expert < 0 || expert >= 256) return -1;\n    return phase6_bundle_slots[layer][expert];\n}\n\nstatic void phase6_read_exact(int fd, void * dst, size_t bytes, size_t offset) {\n    size_t done = 0;\n    const uint64_t start = phase6_now_ns();\n    while (done < bytes) {\n        const ssize_t got = pread(fd, (char *) dst + done, bytes - done, offset + done);\n        if (got < 0) {\n            if (errno == EINTR) continue;\n            phase6_fail("pread failed");\n        }\n        if (got == 0) phase6_fail("short pread");\n        done += (size_t) got;\n    }\n    const uint64_t end = phase6_now_ns();\n    phase6_read_ns += end - start;\n    phase6_read_calls += 3;\n    phase6_read_bytes += bytes;\n    if (phase6_decode_mode) {\n        phase6_dec_read_ns += end - start;\n        phase6_dec_read_bytes += bytes;\n    }\n}\n\nstatic int phase6_reserve_slot(int layer, int expert) {\n    int slot = -1;\n    for (size_t i = 0; i < phase6_slot_count; ++i) {\n        if (!phase6_slots[i].valid) { slot = (int) i; break; }\n    }\n    if (slot < 0) {\n        // JOIN4: victim = oldest NON-PINNED slot (pins never evicted).\n        uint64_t oldest = UINT64_MAX;\n        for (size_t i = 0; i < phase6_slot_count; ++i) {\n            if (!phase6_slots[i].valid) continue;\n            if (phase6_pinned(phase6_slots[i].layer, phase6_slots[i].expert)) continue;\n            if (phase6_slots[i].age < oldest) { oldest = phase6_slots[i].age; slot = (int) i; }\n        }\n        if (slot < 0) {\n            // Degenerate (all slots pinned): evict oldest anyway, counted.\n            oldest = UINT64_MAX;\n            for (size_t i = 0; i < phase6_slot_count; ++i) {\n                if (phase6_slots[i].age < oldest) { oldest = phase6_slots[i].age; slot = (int) i; }\n            }\n            phase6_pin_violations++;\n        }\n        if (phase6_slots[slot].valid) {\n            const int old_layer = phase6_slots[slot].layer;\n            const int old_expert = phase6_slots[slot].expert;\n            phase6_bundle_slots[old_layer][old_expert] = -1;\n            phase6_evictions++;\n            if (phase6_zc_enabled()) {\n                const int ring = phase6_zc_evict_count % PHASE6_ZC_EVICT_RING;\n                phase6_zc_evict_layer[ring] = old_layer;\n                phase6_zc_evict_expert[ring] = old_expert;\n                phase6_zc_evict_count++;\n                phase6_zc_evict_range(old_layer, old_expert);\n            }\n        }\n        /* S2 baseline (noDN): NO madvise(DONTNEED) on staged evict. The refill\n         * pread overwrites the slot pages in place (~15x cheaper per miss,\n         * exact-output; see probes/hw_agent3_memory/REPORT.md). The zero-copy\n         * arm\'s file-range eviction (phase6_zc_evict_range above) is KEEP:\n         * it bounds file-backed RSS, a different mechanism. */\n    }\n    phase6_slots[slot] = (struct phase6_slot) {\n        layer, expert, ++phase6_age, 1,\n        (phase6_async_enabled() && !phase6_zc_enabled()) ? 0 : 7\n    };\n    phase6_bundle_slots[layer][expert] = slot;\n    return slot;\n}\n\n// JOIN4: per-expert accounting with decode split (mode set at prepare entry).\nstatic void phase6_note(int layer, int expert, int hit) {\n    (void) layer; (void) expert;\n    phase6_requests++;\n    if (hit) phase6_hits++; else phase6_misses++;\n    if (phase6_decode_mode) {\n        phase6_dec_requests++;\n        if (hit) phase6_dec_hits++; else phase6_dec_misses++;\n    }\n}\n\nstatic int phase6_load(int layer, int expert) {\n    phase6_init();\n    int slot = phase6_find_slot(layer, expert);\n    if (slot >= 0) {\n        phase6_note(layer, expert, 1);\n        phase6_slots[slot].age = ++phase6_age;\n        return slot;\n    }\n    phase6_note(layer, expert, 0);\n    slot = phase6_reserve_slot(layer, expert);\n    unsigned char * dst = phase6_storage + (size_t) slot * phase6_slot_bytes;\n    size_t in_slot = 0;\n    for (int kind = 0; kind < 3; ++kind) {\n        const struct phase6_record * record = &phase6_records[layer][kind];\n        phase6_read_exact(record->fd, dst + in_slot, record->slice_bytes,\n                          record->base_offset + (size_t) expert * record->slice_bytes);\n        in_slot += record->slice_bytes;\n    }\n    return slot;\n}\n\nstatic void * phase6_async_read_worker(void * opaque) {\n    struct phase6_async_task * task = (struct phase6_async_task *) opaque;\n    const uint64_t start = phase6_now_ns();\n    unsigned char * dst = phase6_storage + (size_t) task->slot * phase6_slot_bytes;\n    size_t in_slot = 0;\n    for (int kind = 0; kind < 3; ++kind) {\n        const struct phase6_record * record = &phase6_records[task->layer][kind];\n        phase6_copy_read(record->fd, dst + in_slot, record->slice_bytes,\n                         record->base_offset + (size_t) task->expert * record->slice_bytes);\n        in_slot += record->slice_bytes;\n        task->read_bytes += record->slice_bytes;\n        __atomic_fetch_or(&phase6_slots[task->slot].ready_mask, 1 << kind, __ATOMIC_RELEASE);\n    }\n    task->read_ns = phase6_now_ns() - start;\n    return NULL;\n}\n\nstatic void phase6_reap_async(void) {\n    if (phase6_async_started_count == 0) return;\n    const uint64_t start = phase6_now_ns();\n    for (int i = 0; i < phase6_async_started_count; ++i) {\n        if (pthread_join(phase6_async_threads[i], NULL) != 0) {\n            phase6_fail("pipeline pthread_join failed");\n        }\n        phase6_read_ns += phase6_async_task_records[i].read_ns;\n        phase6_read_calls += 3;\n        phase6_read_bytes += phase6_async_task_records[i].read_bytes;\n        if (phase6_async_task_records[i].decode) {\n            phase6_dec_read_ns += phase6_async_task_records[i].read_ns;\n            phase6_dec_read_bytes += phase6_async_task_records[i].read_bytes;\n        }\n    }\n    phase6_async_wait_ns += phase6_now_ns() - start;\n    phase6_async_task_count = 0;\n    phase6_async_started_count = 0;\n    phase6_async_layer = -1;\n}\n\nstatic void phase6_prepare_async(const struct ggml_tensor * tensor,\n                                 const struct ggml_tensor * ids) {\n    phase6_init();\n    const int layer = phase6_layer(tensor->name);\n    if (layer < 0 || layer >= 40 || ids == NULL || ids->type != GGML_TYPE_I32) {\n        phase6_fail("invalid async routed node metadata");\n    }\n    // The previous async arm joined every task before compute.  This arm\n    // retains the task records across graph nodes, publishes each plane\'s\n    // readiness, and only reaps after the graph advances to another layer.\n    if (phase6_async_layer >= 0 && phase6_async_layer != layer) phase6_reap_async();\n    if (phase6_async_layer < 0) phase6_async_layer = layer;\n    const int first_new = phase6_async_task_count;\n    for (int64_t row = 0; row < ids->ne[1]; ++row) {\n        for (int64_t i = 0; i < ids->ne[0]; ++i) {\n            const int expert = *(const int32_t *) ((const char *) ids->data +\n                row * ids->nb[1] + i * ids->nb[0]);\n            if (expert < 0 || expert >= 256) phase6_fail("invalid native expert ID");\n            int slot = phase6_find_slot(layer, expert);\n            if (slot >= 0) {\n                phase6_note(layer, expert, 1);\n                phase6_slots[slot].age = ++phase6_age;\n                continue;\n            }\n            phase6_note(layer, expert, 0);\n            slot = phase6_reserve_slot(layer, expert);\n            if (phase6_async_task_count >= PHASE6_MAX_ASYNC_TASKS) phase6_fail("too many pipeline tasks");\n            phase6_async_task_records[phase6_async_task_count++] = (struct phase6_async_task) {\n                layer, expert, slot, 0, 0, phase6_decode_mode\n            };\n        }\n    }\n    for (int i = first_new; i < phase6_async_task_count; ++i) {\n        if (pthread_create(&phase6_async_threads[i], NULL, phase6_async_read_worker,\n                           &phase6_async_task_records[i]) != 0) {\n            phase6_fail("pipeline pthread_create failed");\n        }\n    }\n    phase6_async_started_count = phase6_async_task_count;\n    phase6_async_tasks += (uint64_t) (phase6_async_task_count - first_new);\n}\n\nstatic void phase6_prepare_zc(const struct ggml_tensor * tensor, const struct ggml_tensor * ids) {\n    // Challenger: logical admission only.  No reads, no copies, no threads;\n    // kernels fault the file-backed bytes directly on first touch.\n    phase6_init();\n    const int layer = phase6_layer(tensor->name);\n    if (layer < 0 || layer >= 40 || ids == NULL || ids->type != GGML_TYPE_I32) {\n        phase6_fail("invalid zero-copy routed node metadata");\n    }\n    for (int64_t row = 0; row < ids->ne[1]; ++row) {\n        for (int64_t i = 0; i < ids->ne[0]; ++i) {\n            const int expert = *(const int32_t *) ((const char *) ids->data + row * ids->nb[1] + i * ids->nb[0]);\n            if (expert < 0 || expert >= 256) phase6_fail("invalid native expert ID");\n            int slot = phase6_find_slot(layer, expert);\n            if (slot >= 0) {\n                phase6_note(layer, expert, 1);\n                phase6_slots[slot].age = ++phase6_age;\n                continue;\n            }\n            phase6_note(layer, expert, 0);\n            phase6_reserve_slot(layer, expert);\n        }\n    }\n}\n\nstatic void phase6_prepare_inner(const struct ggml_tensor * tensor, const struct ggml_tensor * ids) {\n    if (phase6_zc_enabled()) {\n        phase6_prepare_zc(tensor, ids);\n        return;\n    }\n    if (phase6_async_enabled()) {\n        phase6_prepare_async(tensor, ids);\n        return;\n    }\n    for (int64_t row = 0; row < ids->ne[1]; ++row) {\n        for (int64_t i = 0; i < ids->ne[0]; ++i) {\n            const int expert = *(const int32_t *) ((const char *) ids->data + row * ids->nb[1] + i * ids->nb[0]);\n            if (expert < 0 || expert >= 256) phase6_fail("invalid native expert ID");\n            phase6_load(phase6_layer(tensor->name), expert);\n        }\n    }\n}\n\nstatic void phase6_prepare(const struct ggml_tensor * tensor, const struct ggml_tensor * ids) {\n    if (!phase6_enabled()) return;\n    const int layer = phase6_layer(tensor->name);\n    if (layer < 0 || layer >= 40 || ids == NULL || ids->type != GGML_TYPE_I32) {\n        phase6_fail("invalid routed node metadata");\n    }\n    // JOIN4: mode + in-node fetch timing (subtracted from expert nodes).\n    phase6_decode_mode = (ids->ne[1] == 1);\n    const uint64_t t0 = phase6_now_ns();\n    phase6_prepare_inner(tensor, ids);\n    const uint64_t dt = phase6_now_ns() - t0;\n    if (join4_profile_enabled()) {\n        if (phase6_decode_mode) join4_dec_fetchprep_ns += dt;\n        else join4_pre_fetchprep_ns += dt;\n    }\n}\n\nstatic const char * phase6_tensor_ptr(const struct ggml_tensor * tensor, int expert) {\n    const int layer = phase6_layer(tensor->name);\n    const int kind = phase6_kind(tensor->name);\n    if (layer < 0 || kind < 0 || !phase6_records[layer][kind].valid) return NULL;\n    const int slot = phase6_find_slot(layer, expert);\n    if (slot < 0) phase6_fail("selected expert was not prepared");\n    if (phase6_zc_enabled()) {\n        // Challenger: consume the exact registered file bytes directly.\n        // Same bytes the control pread-copies; no slot indirection.\n        const struct phase6_record * record = &phase6_records[layer][kind];\n        return (const char *) phase6_zc_file_map + record->base_offset\n            + (size_t) expert * record->slice_bytes;\n    }\n    if (phase6_async_enabled()) {\n        const int want = 1 << kind;\n        const uint64_t start = phase6_now_ns();\n        while ((__atomic_load_n(&phase6_slots[slot].ready_mask, __ATOMIC_ACQUIRE) & want) == 0) {\n#if defined(__linux__)\n            sched_yield();\n#endif\n        }\n        const uint64_t waited = phase6_now_ns() - start;\n        if (waited != 0) {\n            __atomic_fetch_add(&phase6_ready_wait_ns, waited, __ATOMIC_RELAXED);\n            __atomic_fetch_add(&phase6_ready_wait_events, 1, __ATOMIC_RELAXED);\n        }\n    }\n    size_t in_slot = 0;\n    for (int k = 0; k < kind; ++k) in_slot += phase6_records[layer][k].slice_bytes;\n    return (const char *) phase6_storage + (size_t) slot * phase6_slot_bytes + in_slot;\n}\n\n// Keep the native IQP selected-expert path enabled in bounded mode. The\n// storage hook redirects only its source plane; decode and arithmetic remain\n// the exact control implementation.\nconst char * ggml_cpu_phase6_iqp_source(const struct ggml_tensor * tensor, int64_t expert) {\n    if (!phase6_enabled()) return NULL;\n    return phase6_tensor_ptr(tensor, (int) expert);\n}\n\n// JOIN4 section machine: cb-name markers (substring; names are "cb-il").\nenum {\n    JOIN4_M_NONE = 0, JOIN4_M_OPEN_ATTN, JOIN4_M_CLOSE_ATTN, JOIN4_M_CLOSE_GDN,\n    JOIN4_M_OPEN_MOE, JOIN4_M_CLOSE_MOE, JOIN4_M_OPEN_SHARED, JOIN4_M_CLOSE_SHARED,\n    JOIN4_M_OPEN_LMHEAD, JOIN4_M_CLOSE_LMHEAD, JOIN4_M_TAIL\n};\n\nstatic int join4_marker(const char * name) {\n    if (name == NULL) return JOIN4_M_NONE;\n    if (strstr(name, "linear_attn_out")) return JOIN4_M_CLOSE_GDN;\n    if (strstr(name, "attn_output")) return JOIN4_M_CLOSE_ATTN;\n    if (strstr(name, "attn_post_norm")) return JOIN4_M_OPEN_MOE;\n    if (strstr(name, "attn_norm")) return JOIN4_M_OPEN_ATTN;\n    if (strstr(name, "ffn_moe_out")) return JOIN4_M_CLOSE_MOE;\n    if (strstr(name, "ffn_shexp_gated")) return JOIN4_M_CLOSE_SHARED;\n    if (strstr(name, "ffn_shexp")) return JOIN4_M_OPEN_SHARED;\n    if (strstr(name, "result_output")) return JOIN4_M_CLOSE_LMHEAD;\n    if (strstr(name, "result_norm")) return JOIN4_M_OPEN_LMHEAD;\n    if (strstr(name, "ffn_out")) return JOIN4_M_TAIL;\n    if (strstr(name, "l_out")) return JOIN4_M_TAIL;\n    return JOIN4_M_NONE;\n}\n\n// Resolve the ATTN_TENT scratch into ATTN (to_gdn=0) or GDN (to_gdn=1).\nstatic void join4_flush_attn_scratch(int to_gdn) {\n    if (join4_decode_mode) {\n        if (to_gdn) join4_dec_gdn_ns += join4_scratch_dec_ns;\n        else join4_dec_attn_ns += join4_scratch_dec_ns;\n        join4_scratch_dec_ns = 0;\n    } else {\n        if (to_gdn) join4_pre_gdn_ns += join4_scratch_pre_ns;\n        else join4_pre_attn_ns += join4_scratch_pre_ns;\n        join4_scratch_pre_ns = 0;\n    }\n}\n\n// Attribute one node\'s wall time (decode/prefill by current mode).\n// resolve: 0=none, 1=attn, 2=gdn (section closers bypass scratch).\n// wb: weight bucket (-1 = not a matmul; matmuls counted in BOTH).\nstatic void join4_accum(int bucket, int expert, int resolve, int wb,\n                        uint64_t dt) {\n    if (join4_decode_mode) {\n        if (expert) { join4_dec_expert_ns += dt; join4_dec_expert_cnt++; }\n        else if (resolve == 1) join4_dec_attn_ns += dt;\n        else if (resolve == 2) join4_dec_gdn_ns += dt;\n        else if (bucket == JOIN4_SEC_ATTN_TENT) join4_scratch_dec_ns += dt;\n        else { join4_dec_ns[bucket] += dt; join4_dec_cnt[bucket]++; }\n        if (wb >= 0) { join4_dec_wb[wb] += dt; join4_dec_wb_cnt[wb]++; }\n    } else {\n        if (expert) { join4_pre_expert_ns += dt; join4_pre_expert_cnt++; }\n        else if (resolve == 1) join4_pre_attn_ns += dt;\n        else if (resolve == 2) join4_pre_gdn_ns += dt;\n        else if (bucket == JOIN4_SEC_ATTN_TENT) join4_scratch_pre_ns += dt;\n        else { join4_pre_ns[bucket] += dt; join4_pre_cnt[bucket]++; }\n        if (wb >= 0) { join4_pre_wb[wb] += dt; join4_pre_wb_cnt[wb]++; }\n    }\n}\n\n// Force-close any open section (counts surprises; scratch defaults to ATTN).\nstatic void join4_force_close(void) {\n    if (join4_section == JOIN4_SEC_ATTN_TENT) join4_flush_attn_scratch(0);\n    if (join4_section != JOIN4_SEC_MISC) join4_force_closes++;\n    join4_section = JOIN4_SEC_MISC;\n}\n\n// Called on thread 0 at each graph node\'s top (PROFILE only). Flushes the\n// previous pending node ([t0_prev, now) into its bucket: includes the\n// barrier wait, which IS node wall on thread 0), processes markers, sets\n// decode mode at expert nodes, and arms the pending record.\nstatic void join4_node_start(const struct ggml_tensor * node, int node_n) {\n    // Resident arm never runs phase6_init: ensure the atexit report here.\n    if (!phase6_report_registered) {\n        phase6_report_registered = 1;\n        atexit(phase6_report);\n    }\n    const uint64_t now = phase6_now_ns();\n    if (join4_pending_valid) {\n        join4_accum(join4_pending_bucket, join4_pending_expert,\n                    join4_pending_resolve, join4_pending_wb,\n                    now - join4_pending_t0);\n        join4_pending_valid = 0;\n        join4_pending_resolve = 0;\n        join4_pending_wb = -1;\n    }\n    // Nodelist ground truth (first graph only, env-gated).\n    if (!join4_nodelist_done) {\n        if (join4_nodelist_fp == NULL) {\n            const char * nlp = getenv("GGML_PHASE6_NODELIST");\n            if (nlp != NULL && nlp[0] != \'\\0\') {\n                join4_nodelist_fp = fopen(nlp, "w");\n            } else {\n                join4_nodelist_done = 1;\n            }\n        }\n        if (join4_nodelist_fp != NULL && node != NULL) {\n            const char * s0 = (node->src[0] != NULL) ? node->src[0]->name : NULL;\n            fprintf(join4_nodelist_fp, "%d op=%d name=%s src0=%s\\n", node_n,\n                    (int) node->op, node->name ? node->name : "-",\n                    (s0 && s0[0]) ? s0 : "-");\n        }\n    }\n    if (node_n == 0) {\n        join4_graph_moe_nodes = 0;\n        if (!join4_started) {\n            join4_started = 1;\n            join4_prefill_start_ns = now;\n        }\n    }\n    const char * name = (node != NULL) ? node->name : NULL;\n    const int m = join4_marker(name);\n    if (m != JOIN4_M_NONE) join4_markers_seen++;\n    int expert = 0;\n    // Expert nodes: MUL_MAT_ID on _exps weights (time -> EXPERT; the\n    // in-node fetch_prep is subtracted at report; markers still apply).\n    if (node != NULL && node->op == GGML_OP_MUL_MAT_ID && node->src[0] != NULL &&\n        node->src[0]->name != NULL && strstr(node->src[0]->name, "_exps") != NULL) {\n        expert = 1;\n        const struct ggml_tensor * ids = node->src[2];\n        if (ids != NULL && ids->type == GGML_TYPE_I32) {\n            join4_decode_mode = (ids->ne[1] == 1);\n            join4_graph_moe_nodes++;\n        }\n    }\n    int bucket = join4_section;\n    int resolve = 0;\n    switch (m) {\n        case JOIN4_M_OPEN_ATTN:\n            join4_force_close();\n            join4_section = JOIN4_SEC_ATTN_TENT;\n            bucket = JOIN4_SEC_ATTN_TENT;\n            break;\n        case JOIN4_M_CLOSE_ATTN:\n            // Closer attributed directly to ATTN; previous scratch moves too.\n            join4_flush_attn_scratch(0);\n            resolve = 1;\n            join4_section = JOIN4_SEC_MISC;\n            break;\n        case JOIN4_M_CLOSE_GDN:\n            join4_flush_attn_scratch(1);\n            resolve = 2;\n            join4_section = JOIN4_SEC_MISC;\n            break;\n        case JOIN4_M_OPEN_MOE:\n            join4_force_close();\n            join4_section = JOIN4_SEC_MOE;\n            bucket = JOIN4_SEC_MOE;\n            break;\n        case JOIN4_M_CLOSE_MOE:\n            if (join4_section != JOIN4_SEC_MOE && !expert) join4_force_closes++;\n            bucket = JOIN4_SEC_MOE;\n            join4_section = JOIN4_SEC_MISC;\n            break;\n        case JOIN4_M_OPEN_SHARED:\n            join4_force_close();\n            join4_section = JOIN4_SEC_SHARED;\n            bucket = JOIN4_SEC_SHARED;\n            break;\n        case JOIN4_M_CLOSE_SHARED:\n            bucket = JOIN4_SEC_SHARED;\n            join4_section = JOIN4_SEC_MISC;\n            break;\n        case JOIN4_M_OPEN_LMHEAD:\n            join4_force_close();\n            join4_section = JOIN4_SEC_LMHEAD;\n            bucket = JOIN4_SEC_LMHEAD;\n            break;\n        case JOIN4_M_CLOSE_LMHEAD:\n            bucket = JOIN4_SEC_LMHEAD;\n            join4_section = JOIN4_SEC_MISC;\n            break;\n        case JOIN4_M_TAIL:\n            join4_force_close();\n            bucket = JOIN4_SEC_MISC;\n            break;\n        default:\n            break;\n    }\n    if (expert) {\n        join4_pending_bucket = bucket; // unused for expert, kept for debug\n        join4_pending_expert = 1;\n    } else {\n        join4_pending_bucket = bucket;\n        join4_pending_expert = 0;\n    }\n    join4_pending_resolve = expert ? 0 : resolve;\n    // Weight bucket for matmuls (both attributions run).\n    join4_pending_wb = -1;\n    if (node != NULL && (node->op == GGML_OP_MUL_MAT ||\n                         node->op == GGML_OP_MUL_MAT_ID) &&\n        node->src[0] != NULL) {\n        const char * wname = node->src[0]->name;\n        join4_pending_wb = join4_wbucket(wname);\n        if (join4_pending_wb == JOIN4_WB_OTHER &&\n            join4_wb_unmatched_logged < 20 && wname != NULL && wname[0]) {\n            join4_wb_unmatched_logged++;\n            fprintf(stderr, "PHASE6_WB_UNMATCHED %s\\n", wname);\n        }\n    }\n    join4_pending_t0 = now;\n    join4_pending_valid = 1;\n}\n\n// Called on thread 0 at graph loop end (PROFILE only). Flushes the last\n// node, counts decode/prefill graphs, stamps TTFT at first decode end.\nstatic void join4_graph_end(void) {\n    const uint64_t now = phase6_now_ns();\n    if (join4_pending_valid) {\n        join4_accum(join4_pending_bucket, join4_pending_expert,\n                    join4_pending_resolve, join4_pending_wb,\n                    now - join4_pending_t0);\n        join4_pending_valid = 0;\n        join4_pending_resolve = 0;\n        join4_pending_wb = -1;\n    }\n    if (join4_nodelist_fp != NULL) {\n        fclose(join4_nodelist_fp);\n        join4_nodelist_fp = NULL;\n        join4_nodelist_done = 1;\n    }\n    // Hygiene: no section may span graphs (counts surprises, no loss:\n    // every node was already attributed; only scratch re-homes here).\n    if (join4_section == JOIN4_SEC_ATTN_TENT) join4_flush_attn_scratch(0);\n    if (join4_section != JOIN4_SEC_MISC) join4_force_closes++;\n    join4_section = JOIN4_SEC_MISC;\n    if (join4_graph_moe_nodes > 0) {\n        if (join4_decode_mode) {\n            join4_decode_graphs++;\n            if (join4_decode_graphs == 1 && join4_started) {\n                join4_ttft_ns = now - join4_prefill_start_ns;\n            }\n        } else {\n            join4_prefill_graphs++;\n        }\n    }\n}\n\nstatic void ggml_phase6_route_trace(const struct ggml_compute_params * params,\n                                    const struct ggml_tensor * tensor) {\n    if (params->ith != 0 || tensor->op != GGML_OP_MUL_MAT_ID) return;\n    const char * path = getenv("GGML_PHASE6_ROUTE_TRACE");\n    if (path == NULL || path[0] == \'\\0\') return;\n    const struct ggml_tensor * weights = tensor->src[0];\n    const struct ggml_tensor * ids = tensor->src[2];\n    if (weights == NULL || ids == NULL || ids->type != GGML_TYPE_I32 ||\n            strstr(weights->name, "ffn_") == NULL || strstr(weights->name, "_exps") == NULL) return;\n    static FILE * fp = NULL;\n    static uint64_t event_id = 0;\n    if (fp == NULL) {\n        fp = fopen(path, "a");\n        if (fp == NULL) return;\n        setvbuf(fp, NULL, _IOLBF, 0);\n    }\n    const int64_t n = ggml_nelements(ids);\n    const int32_t * values = (const int32_t *) ids->data;\n    fprintf(fp, "{\\"event\\":%" PRIu64 ",\\"weight\\":\\"%s\\",\\"shape\\":[%" PRId64 ",%" PRId64 "],\\"ids\\":[",\n        event_id++, weights->name, ids->ne[0], ids->ne[1]);\n    for (int64_t i = 0; i < n; ++i) fprintf(fp, "%s%d", i ? "," : "", values[i]);\n    fprintf(fp, "]}\\n");\n}\n'

K2_NORM_OLD = '''        ggml_tensor * weights_sum = ggml_sum_rows(ctx0, weights); // [1, n_tokens]
        cb(weights_sum, "ffn_moe_weights_sum", il);'''

K2_NORM_NEW = '''        ggml_tensor * weights_sum = nullptr;
        if (edge0_k2 == edge0_k1) {
            weights_sum = ggml_sum_rows(ctx0, weights); // [1, n_tokens]
        } else {
            // reference mass over top-k2 (paper 2609.04575 Eq.2 denominator)
            ggml_tensor * sel_k2 = ggml_argsort_top_k(ctx0, selection_probs, (int) edge0_k2); // [k2, T]
            ggml_tensor * rk2 = ggml_get_rows(ctx0, probs, sel_k2); // [1, k2, T]
            rk2 = ggml_reshape_2d(ctx0, rk2, edge0_k2, n_tokens); // [k2, T]
            weights_sum = ggml_sum_rows(ctx0, rk2); // [1, T]
        }
        cb(weights_sum, "ffn_moe_weights_sum", il);'''

K1K2_CODE = '''
    // ---- edge0 k1/k2 (env; default native) ----
    int64_t edge0_k1 = n_expert_used;
    int64_t edge0_k2 = n_expert_used;
    if (const char * e1 = getenv("GGML_MOE_K1")) { int v = atoi(e1); if (v > 0) edge0_k1 = v; }
    if (const char * e2 = getenv("GGML_MOE_K2")) { int v = atoi(e2); if (v > 0) edge0_k2 = v; }
    if (edge0_k2 < edge0_k1) edge0_k2 = edge0_k1;
    if (edge0_k2 > n_expert) edge0_k2 = n_expert;
    n_expert_used = edge0_k1;
'''


def run_checked(cmd, cwd=None, log=None):
    print("+", " ".join(str(x) for x in cmd), flush=True)
    p = subprocess.run([str(x) for x in cmd], cwd=cwd, text=True,
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                       check=False)
    print(p.stdout[-3000:], flush=True)
    if log:
        Path(log).write_text(p.stdout, encoding="utf-8")
    if p.returncode:
        raise RuntimeError(f"command exited {p.returncode}: {cmd}")
    return p


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for ch in iter(lambda: f.read(8 << 20), b""):
            h.update(ch)
    return h.hexdigest()


def replace_once(path, old, new):
    text = Path(path).read_text(encoding="utf-8")
    if text.count(old) != 1:
        raise RuntimeError(f"patch anchor mismatch in {path}")
    Path(path).write_text(text.replace(old, new), encoding="utf-8")


def drop_file_cache(path):
    os.sync()
    if not hasattr(os, "posix_fadvise"):
        return {"available": False}
    fd = os.open(path, os.O_RDONLY)
    try:
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        return {"available": True, "called": True}
    except OSError as exc:
        return {"available": True, "called": False, "error": repr(exc)}
    finally:
        os.close(fd)


def proc_sample(pid):
    row = {"mono_ns": time.monotonic_ns(), "rss_kib": 0, "rss_anon_kib": 0,
           "rss_file_kib": 0, "read_bytes": 0, "rchar": 0, "minflt": 0,
           "majflt": 0, "valid": False}
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("VmRSS:"): row["rss_kib"] = int(line.split()[1])
            elif line.startswith("RssAnon:"): row["rss_anon_kib"] = int(line.split()[1])
            elif line.startswith("RssFile:"): row["rss_file_kib"] = int(line.split()[1])
        for line in Path(f"/proc/{pid}/io").read_text().splitlines():
            key, value = line.split(":", 1)
            if key in ("read_bytes", "rchar"): row[key] = int(value.strip())
        stat = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        row["minflt"] = int(stat[7])
        row["majflt"] = int(stat[9])
        row["valid"] = True
    except (FileNotFoundError, ProcessLookupError, PermissionError, IndexError, ValueError):
        pass
    return row


PERF_RE = re.compile(r"eval time\s*=\s*([0-9.]+) ms /\s*([0-9]+) runs\s*\(\s*([0-9.]+) ms per token,\s*([0-9.]+) tokens per second\s*\)")
PREFILL_RE = re.compile(r"prompt eval time\s*=\s*([0-9.]+) ms /\s*([0-9]+) (?:tokens|runs)")
SUMMARY_RE = re.compile(r"\[\s*Prompt:\s*([0-9.]+)\s*t/s\s*\|\s*Generation:\s*([0-9.]+)\s*t/s\s*\]")
CACHE_RE = re.compile(r"PHASE6_BOUNDED_CACHE ([^\n]*)")
PROF_RE = re.compile(r"PHASE6_PROFILE ([^\n]*)")
KV_RE = re.compile(r"([a-z_]+)=([0-9]+)")


def parse_perf(text):
    detailed = PERF_RE.findall(text)
    if detailed:
        elapsed, runs, ms, tps = detailed[-1]
        out = {"ms_per_token": float(ms), "tokens_per_second": float(tps),
               "eval_ms": float(elapsed), "eval_runs": int(runs),
               "perf_source": "detailed"}
    else:
        summary = SUMMARY_RE.findall(text)
        if not summary:
            raise RuntimeError("no parseable llama performance line")
        prompt, generation = summary[-1]
        out = {"prompt_tokens_per_second": float(prompt),
               "tokens_per_second": float(generation),
               "ms_per_token": 1000.0 / float(generation),
               "perf_source": "summary"}
    pre = PREFILL_RE.findall(text)
    if pre:
        ms_p, n_p = pre[-1]
        out.update({"prefill_ms": float(ms_p), "prefill_tokens": int(n_p)})
    return out


def fill_perf_from_prof(perf, prof):
    """Summary-only CLI output lacks eval wall/runs; derive exactly from
    the profiler's decode-graph count (eval_ms = ms/tok x dec_graphs)."""
    if "eval_ms" not in perf:
        perf["eval_ms"] = perf["ms_per_token"] * prof["dec_graphs"]
        perf["eval_runs"] = prof["dec_graphs"]
    return perf


def prefill_c_ms(prof):
    """Prefill wall from C section counters (exact, both arms)."""
    return sum(prof[f"pre_{k}_ns"] for k in
               ("attn", "gdn", "moe_rest", "expert_node", "shared",
                "lmhead", "misc")) / 1e6


def parse_kv_list(text):
    return {k: int(v) for k, v in KV_RE.findall(text)}


def response_payload(stdout, prompt):
    marker = stdout.find("[Start thinking]")
    if marker < 0:
        marker = stdout.rfind(prompt) + len(prompt)
    tail = stdout[marker:]
    perf = SUMMARY_RE.search(tail)
    if perf: tail = tail[:perf.start()]
    if tail.rstrip().endswith("Exiting..."): tail = tail.rstrip()[:-len("Exiting...")]
    return tail.strip()


def setup_runtime():
    SCRATCH.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    if not LLAMA.exists():
        run_checked(["git", "clone", "--filter=blob:none",
                     "https://github.com/ggml-org/llama.cpp.git", str(LLAMA)])
    run_checked(["git", "fetch", "--depth", "1", "origin", LLAMA_COMMIT], cwd=LLAMA)
    run_checked(["git", "checkout", "--detach", LLAMA_COMMIT], cwd=LLAMA)
    head = run_checked(["git", "rev-parse", "HEAD"], cwd=LLAMA).stdout.strip()
    print(f"built base: {head} (want {LLAMA_COMMIT})", flush=True)
    assert head == LLAMA_COMMIT, f"NOT on pin: {head}"
    # K1K2 graph patch (s-kernel/tracek4-proven anchors on this pin)
    g = LLAMA / "src" / "llama-graph.cpp"
    replace_once(g, "#include <cstring>\n#include <numeric>",
                 "#include <cstdlib>\n#include <cstring>\n#include <numeric>")
    replace_once(g,
                 "    ggml_tensor * logits = nullptr;\n\n    if (probs_in == nullptr) {",
                 K1K2_CODE + "\n    ggml_tensor * logits = nullptr;\n\n    if (probs_in == nullptr) {")
    replace_once(g,
                 "    const uint32_t n_expert_used_il = hparams.n_expert_used(il);",
                 "    const uint32_t n_expert_used_il = (uint32_t) n_expert_used; "
                 "// edge0: follows k1 (uniform arch; native-identical unset)")
    replace_once(g, K2_NORM_OLD, K2_NORM_NEW)
    # Lazy expert tensors (Qwen3.6 shares the qwen35moe arch file on this pin)
    qwen = LLAMA / "src" / "models" / "qwen35moe.cpp"
    replace_once(qwen,
                 '        layer.ffn_down_exps = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight", il), { n_ff_exp, n_embd, n_expert }, flags);\n'
                 '        create_tensor_gate_up_exps(layer, il, n_embd, n_ff_exp, n_expert, flags);',
                 '        const int expert_flags = flags | TENSOR_READ_LAZY;\n'
                 '        layer.ffn_down_exps = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight", il), { n_ff_exp, n_embd, n_expert }, expert_flags);\n'
                 '        create_tensor_gate_up_exps(layer, il, n_embd, n_ff_exp, n_expert, expert_flags);')
    # Loader: register lazy tensors with the executor (env-gated)
    loader = LLAMA / "src" / "llama-model-loader.cpp"
    replace_once(loader, "#include <cstring>\n", "#include <cstring>\n#include <cstdlib>\n")
    replace_once(loader, "#include <regex>\n",
                 '#include <regex>\n\nextern "C" void ggml_cpu_phase6_register_lazy_tensor(const struct ggml_tensor *, int, size_t, size_t, size_t);\n')
    replace_once(loader,
                 "        size_t n_size = ggml_nbytes(cur);\n\n        const bool from_mapping = use_mmap || lazy.has(cur);",
                 "        size_t n_size = ggml_nbytes(cur);\n\n"
                 '        if (lazy.has(cur) && std::getenv("GGML_PHASE6_BOUNDED_CACHE") != nullptr) {\n'
                 "            const auto & file = files.at(weight->idx);\n"
                 "            ggml_cpu_phase6_register_lazy_tensor(cur, file->file_id(), weight->offs, n_size, cur->nb[2]);\n"
                 "        }\n\n"
                 "        const bool from_mapping = use_mmap || lazy.has(cur);")
    # ggml-cpu: JOIN4 executor + route trace + MUL_MAT_ID redirect + timers
    cpu = LLAMA / "ggml" / "src" / "ggml-cpu" / "ggml-cpu.c"
    replace_once(cpu, "static struct ggml_state g_state = {0};\n",
                 JOIN4_C + "\nstatic struct ggml_state g_state = {0};\n")
    replace_once(cpu,
                 "    if (tensor->op == GGML_OP_NONE || ggml_is_empty(tensor)) {\n"
                 "        return;\n"
                 "    }\n"
                 "\n"
                 "    // extra_buffer op?\n",
                 "    if (tensor->op == GGML_OP_NONE || ggml_is_empty(tensor)) {\n"
                 "        return;\n"
                 "    }\n"
                 "\n"
                 "    ggml_phase6_route_trace(params, tensor);\n"
                 "\n"
                 "    // extra_buffer op?\n")
    replace_once(cpu,
                 "    const int ith = params->ith;\n"
                 "    const int nth = params->nth;\n\n"
                 "    const enum ggml_type type = src0->type;\n\n"
                 "    const bool src1_cont = ggml_is_contiguous(src1);\n",
                 "    const int ith = params->ith;\n"
                 "    const int nth = params->nth;\n\n"
                 "    const enum ggml_type type = src0->type;\n\n"
                 "    const bool src1_cont = ggml_is_contiguous(src1);\n"
                 "    const bool phase6 = phase6_enabled();\n")
    replace_once(cpu,
                 "    // reset current_chunk\n"
                 "    for (int cur_a = ith; cur_a < n_as; cur_a += nth) {\n"
                 "        atomic_int * current_chunk_ctr = (atomic_int *)(atomic_current_chunk + cur_a);\n"
                 "        *current_chunk_ctr = nth;\n"
                 "    }\n\n"
                 "    ggml_barrier(params->threadpool);\n\n"
                 "    for (int cur_a = 0; cur_a < n_as; ++cur_a) {",
                 "    // reset current_chunk\n"
                 "    for (int cur_a = ith; cur_a < n_as; cur_a += nth) {\n"
                 "        atomic_int * current_chunk_ctr = (atomic_int *)(atomic_current_chunk + cur_a);\n"
                 "        *current_chunk_ctr = nth;\n"
                 "    }\n\n"
                 "    ggml_barrier(params->threadpool);\n\n"
                 "    if (phase6 && ith == 0) phase6_prepare(src0, ids);\n"
                 "    ggml_barrier(params->threadpool);\n\n"
                 "    for (int cur_a = 0; cur_a < n_as; ++cur_a) {")
    replace_once(cpu,
                 "        const char * src0_cur = (const char *) src0->data + cur_a * nb02;\n",
                 "        const char * src0_cur = phase6 ? phase6_tensor_ptr(src0, cur_a)\n"
                 "            : (const char *) src0->data + cur_a * nb02;\n")
    replace_once(cpu,
                 "        // TODO: move fused-op detection into ggml_graph_plan so fusion decisions are made once at planning time\n"
                 "        // Try fused ops, fall back to normal compute\n",
                 "        if (state->ith == 0 && join4_prof_on()) join4_node_start(node, node_n);\n"
                 "        // TODO: move fused-op detection into ggml_graph_plan so fusion decisions are made once at planning time\n"
                 "        // Try fused ops, fall back to normal compute\n")
    replace_once(cpu,
                 "        if (node_n + 1 < cgraph->n_nodes) {\n"
                 "            ggml_barrier(state->threadpool);\n"
                 "        }\n"
                 "    }\n"
                 "\n"
                 "#ifdef GGML_USE_OPENMP",
                 "        if (node_n + 1 < cgraph->n_nodes) {\n"
                 "            ggml_barrier(state->threadpool);\n"
                 "        }\n"
                 "    }\n"
                 "\n"
                 "    if (state->ith == 0 && join4_prof_on()) join4_graph_end();\n"
                 "\n"
                 "#ifdef GGML_USE_OPENMP")
    # iqp: selected-expert source redirect
    iqp = LLAMA / "ggml" / "src" / "ggml-cpu" / "iqp.cpp"
    replace_once(iqp, '#include "iqp.h"\n',
                 '#include "iqp.h"\n\nextern "C" const char * ggml_cpu_phase6_iqp_source(const struct ggml_tensor *, int64_t);\n')
    replace_once(iqp,
                 "    const char * src0_cur = (const char *) src0->data + cur_a * nb02;\n",
                 "    const char * src0_cur = (const char *) src0->data + cur_a * nb02;\n"
                 "    const char * phase6_src0_cur = ggml_cpu_phase6_iqp_source(src0, cur_a);\n"
                 "    if (phase6_src0_cur != NULL) src0_cur = phase6_src0_cur;\n")
    # precise CLI timings
    cli = LLAMA / "tools" / "cli" / "cli-context.cpp"
    replace_once(cli,
                 "[ Prompt: %.1f t/s | Generation: %.1f t/s ]",
                 "[ Prompt: %.6f t/s | Generation: %.6f t/s ]")
    patch = run_checked(["git", "diff"], cwd=LLAMA).stdout
    (OUT / "join4-runtime.patch").write_text(patch, encoding="utf-8")
    return {"llama_commit": LLAMA_COMMIT, "built_head": head,
            "k1": K1, "k2": K2,
            "runtime_patch_sha256": hashlib.sha256(patch.encode()).hexdigest()}


def build():
    run_checked(["cmake", "-S", str(LLAMA), "-B", str(BUILD),
                 "-DCMAKE_BUILD_TYPE=Release", "-DGGML_NATIVE=ON",
                 "-DLLAMA_CURL=ON"],
                log=OUT / "cmake-configure.log")
    run_checked(["cmake", "--build", str(BUILD), "--config", "Release",
                 "-j4", "--target", "llama-cli", "llama-quantize"],
                log=OUT / "cmake-build.log")


def fetch_model():
    m = MODELS[BASE]
    url = f"https://huggingface.co/{m['hf']}/resolve/{m['repo']}/{m['file']}"
    dest = SCRATCH / m["file"]
    if not dest.exists() or dest.stat().st_size != m["size"]:
        run_checked(["curl", "-L", "--fail", "--http1.1", "--retry", "5",
                     "--retry-all-errors", "--retry-delay", "5", "-C", "-",
                     "-o", str(dest), url],
                    log=OUT / "model-download.log")
    d = sha256(dest)
    if dest.stat().st_size != m["size"] or d != m["sha"]:
        raise RuntimeError("model identity mismatch")
    return {"base": BASE, "file": m["file"], "size_bytes": m["size"],
            "sha256": d, "repo_commit": m["repo"]}


GGML_TYPE_NAMES = {
    0: "f32", 1: "f16", 2: "q4_0", 3: "q4_1", 6: "q5_0", 7: "q5_1",
    8: "q8_0", 9: "q8_1", 10: "q2_k", 11: "q3_k", 12: "q4_k",
    13: "q5_k", 14: "q6_k", 15: "q8_k", 16: "iq2_xxs", 17: "iq2_xs",
    18: "iq3_xxs", 19: "iq1_s", 20: "iq4_nl", 21: "iq3_s", 22: "iq2_s",
    23: "iq4_xs", 24: "i8", 25: "i16", 26: "i32", 27: "i64", 28: "f64",
    29: "iq1_m", 30: "bf16", 34: "tq1_0", 35: "tq2_0", 36: "mxfp4"}


def gguf_tensor_table(path):
    import struct as _st
    head = open(path, "rb").read(64 << 20)
    assert head[:4] == b"GGUF"
    n_tensors, n_kv = _st.unpack_from("<QQ", head, 8)
    off = 24

    def read_str(o):
        (n,) = _st.unpack_from("<Q", head, o)
        return head[o + 8:o + 8 + n].decode(), o + 8 + n

    _SCALAR = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1,
               10: 8, 11: 8, 12: 8}

    def skip_value(o, typ):
        if typ in _SCALAR:
            return o + _SCALAR[typ]
        if typ == 8:
            _, o2 = read_str(o)
            return o2
        if typ == 9:
            (at,) = _st.unpack_from("<I", head, o)
            (n,) = _st.unpack_from("<Q", head, o + 4)
            o += 12
            if at == 8:
                for _ in range(n):
                    _, o = read_str(o)
                return o
            return o + {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1,
                        10: 8, 11: 8, 12: 8}[at] * n
        raise RuntimeError(f"kv type {typ}")

    for _ in range(n_kv):
        _, off = read_str(off)
        (typ,) = _st.unpack_from("<I", head, off)
        off = skip_value(off + 4, typ)
    out = []
    for _ in range(n_tensors):
        name, off = read_str(off)
        (nd,) = _st.unpack_from("<I", head, off)
        off += 4 + 8 * nd
        (typ,) = _st.unpack_from("<I", head, off)
        off += 4 + 8
        out.append((name, GGML_TYPE_NAMES.get(typ, f"T{typ}")))
    return out


def transcode_q2k(model_iq2, out_path):
    table = gguf_tensor_table(model_iq2)
    lines, n = [], 0
    for name, typ in table:
        tgt = "q2_k" if "_exps" in name else typ
        n += tgt == "q2_k"
        lines.append(f"^{re.escape(name)}$={tgt}")
    (OUT / "overrides_q2k.txt").write_text("\n".join(lines) + "\n")
    print(f"transcode q2k: {n}/{len(lines)} -> q2_k", flush=True)
    t0 = time.time()
    run_checked([str(QUANTIZE), "--allow-requantize", "--tensor-type-file",
                 str(OUT / "overrides_q2k.txt"), str(model_iq2),
                 str(out_path), "Q8_0", str(THREADS)],
                log=OUT / "transcode_q2k.log")
    dt = time.time() - t0
    os.sync()
    time.sleep(60)
    drop_file_cache(out_path)
    os.sync()
    time.sleep(30)
    chk = gguf_tensor_table(out_path)
    nexp = sum(1 for x, t in chk if "_exps" in x and t == "q2_k")
    if nexp != 120:
        raise RuntimeError(f"q2k expert count {nexp} != 120")
    return {"size_bytes": out_path.stat().st_size,
            "elapsed_sec": dt, "q2k_experts": nexp}


# Sim predictions (tracka_k4_pareto.json) for on-device validation.
EXPECTED = {
    "bounded_3gb": {"hit": 0.5573, "miss_tok": 70.8},
    "bounded_4gb": {"hit": 0.7528, "miss_tok": 39.5},
    "bounded_5gb": {"hit": 0.8590, "miss_tok": 22.6},
    "bounded_6gb": {"hit": 0.9328, "miss_tok": 10.8},
}


def run_case(arm, pid, rep, model, pins_path, tag="", cold=True,
             nodelist=None, n_gen=None):
    """One CLI run. cold=True drops the file cache first (cold-start cost);
    warm runs (cold=False) measure steady state. The executor cache starts
    empty every run (fresh process) either way."""
    name = arm["name"]
    prefix = f"{name}_p{pid:02d}_r{rep}{tag}"
    trace = OUT / f"{prefix}.routes.jsonl"
    stdout_path = OUT / f"{prefix}.stdout.txt"
    stderr_path = OUT / f"{prefix}.stderr.txt"
    samples_path = OUT / f"{prefix}.process.jsonl"
    time_path = OUT / f"{prefix}.time.txt"
    for path in (trace, stdout_path, stderr_path, samples_path, time_path):
        path.unlink(missing_ok=True)
    category, prompt = PROMPTS[pid]
    cli_cmd = [str(CLI), "-m", str(model), "-ngl", "0", "-t", str(THREADS),
               "-c", "512", "-n", str(N_GEN if n_gen is None else n_gen),
               "--temp", "0.7", "--top-p",
               "0.9", "--seed", str(1000 + pid), "--single-turn",
               "--no-display-prompt", "--no-warmup", "--perf", "-lm",
               "mmap", "-lzm", "on" if arm["bounded"] else "off",
               "--poll", "0", "-p", prompt]
    if shutil.which("/usr/bin/time"):
        cmd = ["/usr/bin/time", "-v", "-o", str(time_path)] + cli_cmd
    else:
        cmd = cli_cmd
    env = dict(os.environ, GGML_PHASE6_PROFILE="1",
               GGML_PHASE6_ROUTE_TRACE=str(trace),
               GGML_MOE_K1=str(K1), GGML_MOE_K2=str(K2))
    if nodelist:
        env["GGML_PHASE6_NODELIST"] = str(nodelist)
    if arm["bounded"]:
        env.update(GGML_PHASE6_BOUNDED_CACHE="1",
                   GGML_PHASE6_SLOTS=str(arm["slots"]),
                   GGML_PHASE6_ASYNC="1",
                   GGML_PHASE6_PINS=str(pins_path))
    else:
        for k in ("GGML_PHASE6_BOUNDED_CACHE", "GGML_PHASE6_SLOTS",
                  "GGML_PHASE6_CACHE_BYTES", "GGML_PHASE6_ASYNC",
                  "GGML_PHASE6_PINS", "GGML_PHASE6_ZERO_COPY"):
            env.pop(k, None)
    if cold:
        cache_drop = drop_file_cache(model)
        os.sync()
        time.sleep(SETTLE_SEC)
    else:
        cache_drop = {"available": True, "called": False, "warm": True}
    start = time.monotonic_ns()
    proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True, env=env)
    samples = []
    stop = threading.Event()

    def sample_loop():
        while not stop.is_set():
            samples.append(proc_sample(proc.pid))
            stop.wait(0.02)

    sampler = threading.Thread(target=sample_loop, daemon=True)
    sampler.start()
    out, err = proc.communicate()
    stop.set()
    sampler.join(timeout=2)
    samples.append(proc_sample(proc.pid))
    elapsed = (time.monotonic_ns() - start) / 1e9
    stdout_path.write_text(out, encoding="utf-8")
    stderr_path.write_text(err, encoding="utf-8")
    samples_path.write_text("".join(json.dumps(x) + "\n" for x in samples),
                            encoding="utf-8")
    if proc.returncode:
        raise RuntimeError(f"{prefix} exited {proc.returncode}: {err[-3000:]}")
    valid = [x for x in samples if x.get("valid")]
    perf = parse_perf(out + "\n" + err)
    cache_m = CACHE_RE.findall(err)
    prof_m = PROF_RE.findall(err)
    if not prof_m:
        raise RuntimeError(f"{prefix}: no PHASE6_PROFILE line (timers dead?)")
    prof = parse_kv_list(prof_m[-1])
    cache = parse_kv_list(cache_m[-1]) if cache_m else {}
    fill_perf_from_prof(perf, prof)
    perf["prefill_c_ms"] = prefill_c_ms(prof)
    wb_unmatched = sorted(set(re.findall(r"PHASE6_WB_UNMATCHED (\S+)", err)))
    if arm["bounded"]:
        if not cache_m:
            raise RuntimeError(f"{prefix}: bounded arm has no cache line")
        if cache.get("slots", -1) != arm["slots"]:
            raise RuntimeError(f"{prefix}: slots {cache.get('slots')} != {arm['slots']}")
        exp_pins = {"3.0": 80, "4.0": 80, "5.0": 1346, "6.0": 915}[arm["pins"]]
        if cache.get("pins", -1) != exp_pins:
            raise RuntimeError(f"{prefix}: pins {cache.get('pins')} != {exp_pins}")
        if cache.get("pin_violations", -1) != 0:
            raise RuntimeError(f"{prefix}: pin violations!")
    else:
        if cache_m:
            raise RuntimeError(f"{prefix}: resident arm emitted cache line?!")
    if prof.get("dec_graphs", 0) <= 0:
        raise RuntimeError(f"{prefix}: no decode graphs counted")
    if prof.get("markers", 0) <= 0:
        raise RuntimeError(f"{prefix}: no section markers seen")
    payload = response_payload(out, prompt)
    time_txt = time_path.read_text(encoding="utf-8") if time_path.exists() else ""
    m_rss = re.search(r"Maximum resident set size \(kbytes\): (\d+)", time_txt)
    m_min = re.search(r"Minor \(reclaiming a frame\) page faults: (\d+)", time_txt)
    m_maj = re.search(r"Major \(requiring I/O\) page faults: (\d+)", time_txt)
    return {"arm": name, "pid": pid, "rep": rep, "category": category,
            "cold": cold, "elapsed_sec": elapsed, "cache_drop": cache_drop,
            "peak_rss_mib": max((x["rss_kib"] for x in valid), default=0) / 1024,
            "peak_rss_anon_mib": max((x["rss_anon_kib"] for x in valid), default=0) / 1024,
            "peak_rss_file_mib": max((x["rss_file_kib"] for x in valid), default=0) / 1024,
            "read_bytes_max": max((x["read_bytes"] for x in valid), default=0),
            "rchar_max": max((x["rchar"] for x in valid), default=0),
            "minflt_max": max((x["minflt"] for x in valid), default=0),
            "majflt_max": max((x["majflt"] for x in valid), default=0),
            "time_maxrss_kib": int(m_rss.group(1)) if m_rss else None,
            "time_minflt": int(m_min.group(1)) if m_min else None,
            "time_majflt": int(m_maj.group(1)) if m_maj else None,
            "decode_perf": perf, "cache": cache, "prof": prof,
            "wb_unmatched": wb_unmatched,
            "response_sha256": hashlib.sha256(payload.encode()).hexdigest(),
            "trace_sha256": sha256(trace) if trace.exists() else None}


def cond_summary(runs):
    """Timing/profile means over a run subset (cold or warm)."""
    import statistics as st
    dec_toks = sum(r["prof"]["dec_graphs"] for r in runs)
    dec_wall_ms = sum(r["decode_perf"]["eval_ms"] for r in runs)
    tps_total = dec_toks / (dec_wall_ms / 1000)
    tps_runs = [r["decode_perf"]["tokens_per_second"] for r in runs]
    sec = {}
    for k in ("attn", "gdn", "moe_rest", "expert_node", "fetchprep",
              "shared", "lmhead", "misc"):
        sec[k] = sum(r["prof"][f"dec_{k}_ns"] for r in runs) / dec_toks / 1e6
    sec["expert_compute"] = sec["expert_node"] - sec["fetchprep"]
    sec["router_k2"] = sec["moe_rest"]
    sec["graph_sum"] = sum(sec[k] for k in
                           ("attn", "gdn", "moe_rest", "expert_node",
                            "shared", "lmhead", "misc"))
    wb = {}
    for k in ("exps", "attn", "gdn", "shexp", "router", "out", "other"):
        wb[k] = sum(r["prof"].get(f"wb_dec_{k}", 0) for r in runs) / dec_toks / 1e6
    wb["matmul_sum"] = sum(wb.values())
    return {"runs": len(runs), "dec_toks": dec_toks,
            "tps_total": tps_total, "ms_per_tok": 1000 / tps_total,
            "tps_mean_runs": st.mean(tps_runs),
            "tps_stdev_runs": st.stdev(tps_runs) if len(tps_runs) > 1 else 0.0,
            "sections_ms_tok": sec, "wb_ms_tok": wb,
            "ttft_ms_mean": st.mean(r["prof"]["ttft_ns"] / 1e6 for r in runs),
            "prefill_c_ms_mean": st.mean(r["decode_perf"]["prefill_c_ms"] for r in runs)}


def arm_summary(name, runs):
    warm = [r for r in runs if not r["cold"]]
    cold = [r for r in runs if r["cold"]]
    out = {"warm": cond_summary(warm) if warm else None,
           "cold": cond_summary(cold) if cold else None,
           # Headline = warm steady state (flattened for convenience)
           "runs": len(runs),
           "dec_toks": sum(r["prof"]["dec_graphs"] for r in runs),
           "peak_rss_mib_max": max(r["peak_rss_mib"] for r in runs),
           "peak_rss_anon_mib_max": max(r["peak_rss_anon_mib"] for r in runs),
           "peak_rss_file_mib_max": max(r["peak_rss_file_mib"] for r in runs),
           "time_maxrss_kib_max": max((r["time_maxrss_kib"] or 0 for r in runs), default=None),
           "read_bytes_total": sum(r["read_bytes_max"] for r in runs),
           "minflt_max": max(r["minflt_max"] for r in runs),
           "majflt_max": max(r["majflt_max"] for r in runs)}
    w = out["warm"] or out["cold"]
    out.update({"tps_total": w["tps_total"], "ms_per_tok": w["ms_per_tok"],
                "sections_ms_tok": w["sections_ms_tok"],
                "wb_ms_tok": w["wb_ms_tok"],
                "ttft_ms_mean": w["ttft_ms_mean"],
                "prefill_c_ms_mean": w["prefill_c_ms_mean"]})
    if runs[0]["cache"]:
        # Hit rate is file-temp independent (executor cache always starts
        # empty); aggregate over ALL runs for max N.
        req = sum(r["cache"]["dec_requests"] for r in runs)
        hit = sum(r["cache"]["dec_hits"] for r in runs)
        mis = sum(r["cache"]["dec_misses"] for r in runs)
        dec_toks = out["dec_toks"]
        out["cache"] = {
            "hit_rate": hit / req, "miss_tok": mis / dec_toks,
            "bytes_tok": sum(r["cache"]["dec_read_bytes"] for r in runs) / dec_toks,
            "read_ns_tok": sum(r["cache"]["read_ns"] for r in runs) / dec_toks / 1e6,
            "async_wait_ms_tok": sum(r["cache"]["async_wait_ns"] for r in runs) / dec_toks / 1e6,
            "ready_wait_ms_tok": sum(r["cache"]["ready_wait_ns"] for r in runs) / dec_toks / 1e6,
            "evictions": sum(r["cache"]["evictions"] for r in runs),
            "preload_bytes": runs[0]["cache"]["preload_bytes"],
            "slot_bytes": runs[0]["cache"]["slot_bytes"]}
    return out


def main():
    started = time.time()
    runtime = setup_runtime()
    wfree = shutil.disk_usage(WORK).free / 1e9
    sfree = shutil.disk_usage(SCRATCH).free / 1e9
    print(f"disk free: WORK {wfree:.1f}GB SCRATCH {sfree:.1f}GB", flush=True)
    if wfree < 15 or sfree < 15:
        raise RuntimeError(f"disk too tight: WORK {wfree} SCRATCH {sfree}")
    build()
    model = fetch_model()
    hardware = {"platform": platform.platform(), "cpu_count": os.cpu_count(),
                "threads_used": THREADS,
                "cpuinfo": Path("/proc/cpuinfo").read_text()[:6000],
                "meminfo": Path("/proc/meminfo").read_text()}
    (OUT / "hardware.json").write_text(json.dumps(hardware, indent=2))
    model_iq2 = SCRATCH / MODELS[BASE]["file"]
    q2k_path = WORK / Q2K_NAME
    transcode = transcode_q2k(model_iq2, q2k_path)
    model["q2k_file"] = Q2K_NAME
    model["transcode"] = transcode
    # Locked pins to files (counts asserted here + in C at runtime)
    pins_map = {}
    for tag, blob, n in (("3.0", PINS_3, 80), ("4.0", PINS_4, 80),
                         ("5.0", PINS_5, 1346), ("6.0", PINS_6, 915)):
        p = OUT / f"pins_{tag}.txt"
        p.write_text(blob if blob.endswith("\n") else blob + "\n")
        keys = [x for x in p.read_text().split() if x.strip()]
        assert len(keys) == n, (tag, len(keys), n)
        assert all(0 <= int(k) < 10240 for k in keys), tag
        pins_map[tag] = p
    # SMOKE: resident + locked arm on one prompt; gates the full loop
    print("===== SMOKE (resident vs bounded_6gb, pid 21) =====", flush=True)
    res = next(a for a in ARMS if a["name"] == "resident")
    b6 = next(a for a in ARMS if a["name"] == "bounded_6gb")
    s0 = run_case(res, 21, 0, q2k_path, None, tag="_smoke")
    s1 = run_case(b6, 21, 0, q2k_path, pins_map["6.0"], tag="_smoke")
    print(f"  smoke resident: {s0['decode_perf']['tokens_per_second']:.2f} t/s "
          f"rss={s0['peak_rss_mib']:.0f}MiB", flush=True)
    print(f"  smoke b6: {s1['decode_perf']['tokens_per_second']:.2f} t/s "
          f"rss={s1['peak_rss_mib']:.0f}MiB hit={s1['cache']['dec_hits']/s1['cache']['dec_requests']:.4f} "
          f"markers={s1['prof']['markers']}", flush=True)
    if s0["response_sha256"] != s1["response_sha256"]:
        raise RuntimeError("SMOKE FAIL: bounded output != resident output")
    if s0["trace_sha256"] != s1["trace_sha256"]:
        raise RuntimeError("SMOKE FAIL: bounded routes != resident routes")
    print("  smoke: bit-exact outputs + routes OK", flush=True)
    # NODELIST ground truth (one short resident run; first graph only)
    print("===== NODELIST (resident, pid 21, n=1) =====", flush=True)
    nl = run_case(res, 21, 0, q2k_path, None, tag="_nodelist", cold=False,
                  nodelist=OUT / "nodelist.txt", n_gen=1)
    print(f"  nodelist nodes: {len((OUT / 'nodelist.txt').read_text().splitlines())}",
          flush=True)
    # FULL LOOP (first run per arm cold, rest warm; the file stays
    # warm within an arm after the first drop)
    runs = []
    for arm in ARMS:
        for pid in PIDS:
            for rep in range(1, arm["reps"] + 1):
                cold = (pid == PIDS[0] and rep == 1)
                print(f"===== {arm['name']} pid={pid:02d} rep={rep} "
                      f"{'cold' if cold else 'warm'} =====", flush=True)
                r = run_case(arm, pid, rep, q2k_path,
                             pins_map.get(arm.get("pins", "")), cold=cold)
                runs.append(r)
                c = r["cache"]
                extra = (f" hit={c['dec_hits']/c['dec_requests']:.3f}"
                         if c else "")
                print(f"  {r['decode_perf']['tokens_per_second']:.2f} t/s "
                      f"rss={r['peak_rss_mib']:.0f}MiB{extra}", flush=True)
    # Bit-exactness gate: every pid has ONE output across arms+reps
    sha_gate = {}
    for pid in PIDS:
        shas = set(r["response_sha256"] for r in runs if r["pid"] == pid)
        sha_gate[pid] = len(shas)
        if len(shas) != 1:
            raise RuntimeError(f"pid {pid}: {len(shas)} distinct outputs!")
    # Weight-bucket coverage gate: unmatched matmul weights must be ~none
    unmatched = sorted(set(u for r in runs for u in r["wb_unmatched"]))
    print(f"  wb unmatched weights: {unmatched if unmatched else 'NONE'}",
          flush=True)
    summaries = {}
    for arm in ARMS:
        summaries[arm["name"]] = arm_summary(
            arm["name"], [r for r in runs if r["arm"] == arm["name"]])
    # Sim validation: measured vs predicted hit/miss per bounded arm
    sim_check = {}
    for name, exp in EXPECTED.items():
        if name not in summaries:
            continue
        m = summaries[name]["cache"]
        sim_check[name] = {
            "pred_hit": exp["hit"], "meas_hit": m["hit_rate"],
            "pred_miss_tok": exp["miss_tok"], "meas_miss_tok": m["miss_tok"]}
    q2k_path.unlink()  # keep the pull small (telemetry + logs only)
    result = {"schema": "native-sparse-edge0join4b/v1", "status": "ok",
              "wb_unmatched": unmatched,
              "runtime": runtime, "model": model,
              "hardware": {k: v for k, v in hardware.items()
                           if k not in ("cpuinfo", "meminfo")},
              "decode": {"k1": K1, "k2": K2, "threads": THREADS,
                         "tokens_requested_per_prompt": N_GEN,
                         "pids": PIDS},
              "arms": [a["name"] for a in ARMS],
              "summaries": summaries, "sim_check": sim_check,
              "sha_gate": sha_gate, "wall_sec": time.time() - started}
    (OUT / "result.json").write_text(json.dumps(result, indent=2))
    print("== FINAL means ==", flush=True)
    for name, s in summaries.items():
        c = s.get("cache", {})
        print(f"{name:12s} {s['tps_total']:.2f} t/s ({s['ms_per_tok']:.1f} ms/tok) "
              f"rss={s['peak_rss_mib_max']:.0f}MiB"
              + (f" hit={c['hit_rate']:.4f} miss/t={c['miss_tok']:.1f} "
                 f"B/t={c['bytes_tok']:.0f}" if c else ""), flush=True)
    print(json.dumps({"status": "ok", "summaries": summaries,
                      "sim_check": sim_check}), flush=True)


if __name__ == "__main__":
    main()
