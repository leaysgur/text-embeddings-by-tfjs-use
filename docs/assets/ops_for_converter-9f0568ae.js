function qo(e,t){for(var n=0;n<t.length;n++){const s=t[n];if(typeof s!="string"&&!Array.isArray(s)){for(const r in s)if(r!=="default"&&!(r in e)){const o=Object.getOwnPropertyDescriptor(s,r);o&&Object.defineProperty(e,r,o.get?o:{enumerable:!0,get:()=>s[r]})}}}return Object.freeze(Object.defineProperty(e,Symbol.toStringTag,{value:"Module"}))}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Ko=1e-7,zo=1e-4;class Jw{constructor(t,n){this.backend=t,this.dataMover=n,this.data=new WeakMap,this.dataIdsCount=0}get(t){return this.data.has(t)||this.dataMover.moveData(this.backend,t),this.data.get(t)}set(t,n){this.dataIdsCount++,this.data.set(t,n)}has(t){return this.data.has(t)}delete(t){return this.dataIdsCount--,this.data.delete(t)}numDataIds(){return this.dataIdsCount}}class Go{refCount(t){return ut("refCount")}incRef(t){return ut("incRef")}timerAvailable(){return!0}time(t){return ut("time")}read(t){return ut("read")}readSync(t){return ut("readSync")}readToGPU(t,n){return ut("readToGPU")}numDataIds(){return ut("numDataIds")}disposeData(t,n){return ut("disposeData")}write(t,n,s){return ut("write")}move(t,n,s,r,o){return ut("move")}createTensorFromGPUData(t,n,s){return ut("createTensorFromGPUData")}memory(){return ut("memory")}floatPrecision(){return ut("floatPrecision")}epsilon(){return this.floatPrecision()===32?Ko:zo}dispose(){return ut("dispose")}}function ut(e){throw new Error(`'${e}' not yet implemented or not found in the registry. This kernel may not be supported by the tfjs backend you have chosen`)}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function er(e){let t=e.length,n=0;for(;t>0;)n=Math.random()*t|0,t--,sn(e,t,n)}function Ho(e,t){if(e.length!==t.length)throw new Error(`Array sizes must match to be shuffled together First array length was ${e.length}Second array length was ${t.length}`);let n=e.length,s=0;for(;n>0;)s=Math.random()*n|0,n--,sn(e,n,s),sn(t,n,s)}function Vo(e,t,n){return Math.max(e,Math.min(t,n))}function jo(e){return e%2===0?e:e+1}function sn(e,t,n){const s=e[t];e[t]=e[n],e[n]=s}function Xo(e){let t=0;for(let n=0;n<e.length;n++)t+=e[n];return t}function Yo(e,t){const n=Math.random();return t*n+(1-n)*e}function Zo(e,t){let n=0;for(let s=0;s<e.length;s++){const r=Number(e[s])-Number(t[s]);n+=r*r}return n}function p(e,t){if(!e)throw new Error(typeof t=="string"?t:t())}function at(e,t,n=""){p(Wt(e,t),()=>n+` Shapes ${e} and ${t} must match`)}function re(e){p(e!=null,()=>"The input to the tensor constructor must be a non-null value.")}function Z(e){if(e.length===0)return 1;let t=e[0];for(let n=1;n<e.length;n++)t*=e[n];return t}function Jo(e){return e.length===0}function nr(e,t){if(e===t)return!0;if(e==null||t==null||e.length!==t.length)return!1;for(let n=0;n<e.length;n++)if(e[n]!==null&&t[n]!==null&&e[n]!==t[n])return!1;return!0}function Wt(e,t){if(e===t)return!0;if(e==null||t==null||e.length!==t.length)return!1;for(let n=0;n<e.length;n++)if(e[n]!==t[n])return!1;return!0}function $e(e){return e%1===0}function Qo(e){if(Math.tanh!=null)return Math.tanh(e);if(e===1/0)return 1;if(e===-1/0)return-1;{const t=Math.exp(2*e);return(t-1)/(t+1)}}function ta(e){const t=Math.ceil(Math.sqrt(e));return[t,Math.ceil(e/t)]}function ea(e){const t=new Uint32Array(e);for(let n=0;n<e;++n)t[n]=n;return er(t),t}function Ae(e,t){return t<=e.length?e:e+" ".repeat(t-e.length)}function na(e,t=r=>0,n,s){return new Promise((r,o)=>{let a=0;const i=()=>{if(e()){r();return}a++;const c=t(a);if(n!=null&&a>=n){o();return}s!=null?s(i,c):setTimeout(i,c)};i()})}function sa(e,t){let n=1,s=-1;for(let o=0;o<e.length;++o)if(e[o]>=0)n*=e[o];else if(e[o]===-1){if(s!==-1)throw Error(`Shapes can only have 1 implicit size. Found -1 at dim ${s} and dim ${o}`);s=o}else if(e[o]<0)throw Error(`Shapes can not be < 0. Found ${e[o]} at dim ${o}`);if(s===-1){if(t>0&&t!==n)throw Error(`Size(${t}) must match the product of shape ${e}`);return e}if(n===0)throw Error(`Cannot infer the missing size in [${e}] when there are 0 elements`);if(t%n!==0)throw Error(`The implicit shape can't be a fractional number. Got ${t} / ${n}`);const r=e.slice();return r[s]=t/n,r}function Ge(e,t){const n=t.length;return e=e==null?t.map((s,r)=>r):[].concat(e),p(e.every(s=>s>=-n&&s<n),()=>`All values in axis param must be in range [-${n}, ${n}) but got axis ${e}`),p(e.every(s=>$e(s)),()=>`All values in axis param must be integers but got axis ${e}`),e.map(s=>s<0?n+s:s)}function sr(e,t){const n=[],s=[],r=t!=null&&Array.isArray(t)&&t.length===0,o=t==null||r?null:Ge(t,e).sort();let a=0;for(let i=0;i<e.length;++i){if(o!=null){if(o[a]===i&&e[i]!==1)throw new Error(`Can't squeeze axis ${i} since its dim '${e[i]}' is not 1`);(o[a]==null||o[a]>i)&&e[i]===1&&(n.push(e[i]),s.push(i)),o[a]<=i&&a++}e[i]!==1&&(n.push(e[i]),s.push(i))}return{newShape:n,keptDims:s}}function rr(e,t){return Qn(e,t)}function Qn(e,t){let n=null;if(e==null||e==="float32")n=new Float32Array(t);else if(e==="int32")n=new Int32Array(t);else if(e==="bool")n=new Uint8Array(t);else if(e==="string")n=new Array(t);else throw new Error(`Unknown data type ${e}`);return n}function or(e,t){for(let n=0;n<e.length;n++){const s=e[n];if(isNaN(s)||!isFinite(s))throw Error(`A tensor of type ${t} being uploaded contains ${s}.`)}}function ar(e){return e==="bool"||e==="complex64"||e==="float32"||e==="int32"||e==="string"}function ra(e,t){return!(t==="complex64"||t==="float32"&&e!=="complex64"||t==="int32"&&e!=="float32"&&e!=="complex64"||t==="bool"&&e==="bool")}function rn(e){if(e==="float32"||e==="int32")return 4;if(e==="complex64")return 8;if(e==="bool")return 1;throw new Error(`Unknown dtype ${e}`)}function ir(e){if(e==null)return 0;let t=0;return e.forEach(n=>t+=n.length),t}function pn(e){return typeof e=="string"||e instanceof String}function cr(e){return typeof e=="boolean"}function ur(e){return typeof e=="number"}function He(e){return Array.isArray(e)?He(e[0]):e instanceof Float32Array?"float32":e instanceof Int32Array||e instanceof Uint8Array||e instanceof Uint8ClampedArray?"int32":ur(e)?"float32":pn(e)?"string":cr(e)?"bool":"float32"}function Lt(e){return!!(e&&e.constructor&&e.call&&e.apply)}function oa(e,t){for(let n=t;n<e;++n)if(e%n===0)return n;return e}function Ve(e){const t=e.length;if(t<2)return[];const n=new Array(t-1);n[t-2]=e[t-1];for(let s=t-3;s>=0;--s)n[s]=n[s+1]*e[s+1];return n}function lr(e,t,n,s=!1){const r=new Array;if(t.length===1){const o=t[0]*(s?2:1);for(let a=0;a<o;a++)r[a]=n[e+a]}else{const o=t[0],a=t.slice(1),i=a.reduce((c,u)=>c*u)*(s?2:1);for(let c=0;c<o;c++)r[c]=lr(e+c*i,a,n,s)}return r}function pe(e,t,n=!1){if(e.length===0)return t[0];const s=e.reduce((r,o)=>r*o)*(n?2:1);if(s===0)return[];if(s!==t.length)throw new Error(`[${e}] does not match the input size ${t.length}${n?" for a complex tensor":""}.`);return lr(0,e,t,n)}function aa(e,t){if(Array.isArray(e))return e;if(t==="float32")return e instanceof Float32Array?e:new Float32Array(e);if(t==="int32")return e instanceof Int32Array?e:new Int32Array(e);if(t==="bool"||t==="string")return Uint8Array.from(new Int32Array(e));throw new Error(`Unknown dtype ${t}`)}function ts(e,t){const n=gn(e,t);for(let s=0;s<n.length;s++)n[s]=1;return n}function gn(e,t){if(t==null||t==="float32"||t==="complex64")return new Float32Array(e);if(t==="int32")return new Int32Array(e);if(t==="bool")return new Uint8Array(e);throw new Error(`Unknown data type ${t}`)}function ia(e,t){const n=e.reduce((s,r)=>s*r,1);if(t==null||t==="float32")return pe(e,new Float32Array(n));if(t==="int32")return pe(e,new Int32Array(n));if(t==="bool")return pe(e,new Uint8Array(n));throw new Error(`Unknown data type ${t}`)}function dt(e){e.forEach(t=>{p(Number.isInteger(t)&&t>=0,()=>`Tensor must have a shape comprised of positive integers but got shape [${e}].`)})}function ca(e,t,n){if(t===0)return 0;if(t===1)return e[0];let s=e[e.length-1];for(let r=0;r<e.length-1;++r)s+=n[r]*e[r];return s}function ua(e,t,n){if(t===0)return[];if(t===1)return[e];const s=new Array(t);for(let r=0;r<s.length-1;++r)s[r]=Math.floor(e/n[r]),e-=s[r]*n[r];return s[s.length-1]=e,s}function mn(e){return e&&e.then&&typeof e.then=="function"}/**
 * @license
 * Copyright 2017 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Fs="tfjsflags";class la{constructor(t){this.global=t,this.flags={},this.flagRegistry={},this.urlFlags={},this.getQueryParams=ha,this.populateURLFlags()}setPlatform(t,n){this.platform!=null&&(B().getBool("IS_TEST")||B().getBool("PROD")||console.warn(`Platform ${this.platformName} has already been set. Overwriting the platform with ${t}.`)),this.platformName=t,this.platform=n}registerFlag(t,n,s){if(this.flagRegistry[t]={evaluationFn:n,setHook:s},this.urlFlags[t]!=null){const r=this.urlFlags[t];B().getBool("IS_TEST")||B().getBool("PROD")||console.warn(`Setting feature override from URL ${t}: ${r}.`),this.set(t,r)}}async getAsync(t){return t in this.flags?this.flags[t]:(this.flags[t]=await this.evaluateFlag(t),this.flags[t])}get(t){if(t in this.flags)return this.flags[t];const n=this.evaluateFlag(t);if(mn(n))throw new Error(`Flag ${t} cannot be synchronously evaluated. Please use getAsync() instead.`);return this.flags[t]=n,this.flags[t]}getNumber(t){return this.get(t)}getBool(t){return this.get(t)}getString(t){return this.get(t)}getFlags(){return this.flags}get features(){return this.flags}set(t,n){if(this.flagRegistry[t]==null)throw new Error(`Cannot set flag ${t} as it has not been registered.`);this.flags[t]=n,this.flagRegistry[t].setHook!=null&&this.flagRegistry[t].setHook(n)}evaluateFlag(t){if(this.flagRegistry[t]==null)throw new Error(`Cannot evaluate flag '${t}': no evaluation function found.`);return this.flagRegistry[t].evaluationFn()}setFlags(t){this.flags=Object.assign({},t)}reset(){this.flags={},this.urlFlags={},this.populateURLFlags()}populateURLFlags(){if(typeof this.global>"u"||typeof this.global.location>"u"||typeof this.global.location.search>"u")return;const t=this.getQueryParams(this.global.location.search);Fs in t&&t[Fs].split(",").forEach(s=>{const[r,o]=s.split(":");this.urlFlags[r]=da(r,o)})}}function ha(e){const t={};return e.replace(/[?&]([^=?&]+)(?:=([^&]*))?/g,(n,...s)=>(fa(t,s[0],s[1]),s.join("="))),t}function fa(e,t,n){e[decodeURIComponent(t)]=decodeURIComponent(n||"")}function da(e,t){const n=t.toLowerCase();return n==="true"||n==="false"?n==="true":`${+n}`===n?+n:t}function B(){return hr}let hr=null;function pa(e){hr=e}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */let _n;function fr(){if(_n==null){let e;if(typeof window<"u")e=window;else if(typeof global<"u")e=global;else if(typeof process<"u")e=process;else if(typeof self<"u")e=self;else throw new Error("Could not find a global object");_n=e}return _n}function ga(){const e=fr();return e._tfGlobals==null&&(e._tfGlobals=new Map),e._tfGlobals}function es(e,t){const n=ga();if(n.has(e))return n.get(e);{const s=t();return n.set(e,s),n.get(e)}}const ma="Abs",ba="Acos",wa="Acosh",dr="Add",ya="AddN",$a="All",xa="Any",ka="ArgMax",va="ArgMin",Ea="Asin",Sa="Asinh",Ta="Atan",_a="Atanh",Ia="Atan2",Da="AvgPool",Qw="AvgPoolGrad",Na="AvgPool3D",t1="AvgPool3DGrad",Aa="BatchMatMul",Ma="BatchToSpaceND",Fa="Bincount",Ba="BitwiseAnd",e1="BroadcastTo",Ca="BroadcastArgs",pr="Cast",Ra="Ceil",Pa="ClipByValue",Oa="Complex",La="ComplexAbs",Ua="Concat",Wa="Conv2D",qa="Conv2DBackpropFilter",Ka="Conv2DBackpropInput",za="Conv3D",n1="Conv3DBackpropFilterV2",Ga="Conv3DBackpropInputV2",Ha="Cos",Va="Cosh",ja="Cumprod",Xa="Cumsum",Ya="CropAndResize",Za="DenseBincount",Ja="DepthToSpace",Qa="DepthwiseConv2dNative",ti="DepthwiseConv2dNativeBackpropFilter",ei="DepthwiseConv2dNativeBackpropInput",ni="Diag",si="Dilation2D",s1="Dilation2DBackpropInput",r1="Dilation2DBackpropFilter",o1="Draw",ri="RealDiv",oi="Einsum",ai="Elu",a1="EluGrad",ii="Erf",ci="Equal",ui="Exp",li="ExpandDims",hi="Expm1",fi="FFT",di="Fill",pi="FlipLeftRight",gi="Floor",mi="FloorDiv",bi="FusedBatchNorm",wi="GatherV2",yi="GatherNd",$i="Greater",xi="GreaterEqual",gr="Identity",ki="IFFT",vi="Imag",Ei="IsFinite",Si="IsInf",Ti="IsNan",_i="LeakyRelu",Ii="Less",Di="LessEqual",Ni="LinSpace",Ai="Log",Mi="Log1p",Fi="LogicalAnd",Bi="LogicalNot",Ci="LogicalOr",i1="LogicalXor",c1="LogSoftmax",u1="LowerBound",Ri="LRN",l1="LRNGrad",h1="MatrixBandPart",Pi="Max",Oi="Maximum",Li="MaxPool",f1="MaxPoolGrad",Ui="MaxPool3D",d1="MaxPool3DGrad",Wi="MaxPoolWithArgmax",qi="Mean",Ki="Min",zi="Minimum",Gi="MirrorPad",Hi="Mod",Vi="Multinomial",ji="Multiply",Xi="Neg",Yi="NotEqual",Zi="NonMaxSuppressionV3",Ji="NonMaxSuppressionV4",Qi="NonMaxSuppressionV5",tc="OnesLike",ec="OneHot",nc="Pack",sc="PadV2",p1="Pool",rc="Pow",oc="Prelu",ac="Prod",ic="RaggedGather",cc="RaggedRange",uc="RaggedTensorToTensor",lc="Range",hc="Real",fc="Reciprocal",dc="Relu",pc="Reshape",gc="ResizeNearestNeighbor",g1="ResizeNearestNeighborGrad",mc="ResizeBilinear",m1="ResizeBilinearGrad",bc="Relu6",wc="Reverse",yc="Round",$c="Rsqrt",xc="ScatterNd",kc="TensorScatterUpdate",vc="SearchSorted",Ec="Select",Sc="Selu",Tc="Slice",_c="Sin",Ic="Sinh",Dc="Sign",Nc="Sigmoid",Ac="Softplus",Mc="Sqrt",Fc="Sum",Bc="SpaceToBatchND",Cc="SplitV",Rc="Softmax",Pc="SparseFillEmptyRows",Oc="SparseReshape",Lc="SparseSegmentMean",Uc="SparseSegmentSum",Wc="SparseToDense",qc="SquaredDifference",b1="Square",Kc="StaticRegexReplace",zc="StridedSlice",Gc="StringNGrams",Hc="StringSplit",Vc="StringToHashBucketFast",jc="Sub",Xc="Tan",Yc="Tanh",mr="Tile",Zc="TopK",Jc="Transform",In="Transpose",Qc="Unique",tu="Unpack",eu="UnsortedSegmentSum",w1="UpperBound",nu="ZerosLike",su="Step",y1="FromPixels",ru="RotateWithOffset",Bs="_FusedMatMul",Cs="FusedConv2D",Rs="FusedDepthwiseConv2D";/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Vt(...e){B().getBool("IS_TEST")||B().getBool("PROD")||console.warn(...e)}function $1(...e){B().getBool("IS_TEST")||B().getBool("PROD")||console.log(...e)}/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const xe=es("kernelRegistry",()=>new Map),Be=es("gradRegistry",()=>new Map);function Ps(e,t){const n=ns(e,t);return xe.get(n)}function Os(e){return Be.get(e)}function Fn(e){const t=xe.entries(),n=[];for(;;){const{done:s,value:r}=t.next();if(s)break;const[o,a]=r,[i]=o.split("_");i===e&&n.push(a)}return n}function ou(e){const{kernelName:t,backendName:n}=e,s=ns(t,n);xe.has(s)&&Vt(`The kernel '${t}' for backend '${n}' is already registered`),xe.set(s,e)}function x1(e){const{kernelName:t}=e;Be.has(t)&&B().getBool("DEBUG")&&Vt(`Overriding the gradient for '${t}'`),Be.set(t,e)}function k1(e,t){const n=ns(e,t);if(!xe.has(n))throw new Error(`The kernel '${e}' for backend '${t}' is not registered`);xe.delete(n)}function v1(e){if(!Be.has(e))throw new Error(`The gradient '${e}' for backend is not registered`);Be.delete(e)}function E1(e,t){Fn(e).forEach(s=>{const r=Object.assign({},s,{backendName:t});ou(r)})}function ns(e,t){return`${t}_${e}`}/**
 * @license
 * Copyright 2023 Google LLC.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function br(e){return e instanceof Float32Array||e instanceof Int32Array||e instanceof Uint8Array||e instanceof Uint8ClampedArray}var oe=typeof globalThis<"u"?globalThis:typeof window<"u"?window:typeof global<"u"?global:typeof self<"u"?self:{};function au(e){return e&&e.__esModule&&Object.prototype.hasOwnProperty.call(e,"default")?e.default:e}function iu(e){if(e.__esModule)return e;var t=e.default;if(typeof t=="function"){var n=function s(){if(this instanceof s){var r=[null];r.push.apply(r,arguments);var o=Function.bind.apply(t,r);return new o}return t.apply(this,arguments)};n.prototype=t.prototype}else n={};return Object.defineProperty(n,"__esModule",{value:!0}),Object.keys(e).forEach(function(s){var r=Object.getOwnPropertyDescriptor(e,s);Object.defineProperty(n,s,r.get?r:{enumerable:!0,get:function(){return e[s]}})}),n}var wr=G,yt=null;try{yt=new WebAssembly.Instance(new WebAssembly.Module(new Uint8Array([0,97,115,109,1,0,0,0,1,13,2,96,0,1,127,96,4,127,127,127,127,1,127,3,7,6,0,1,1,1,1,1,6,6,1,127,1,65,0,11,7,50,6,3,109,117,108,0,1,5,100,105,118,95,115,0,2,5,100,105,118,95,117,0,3,5,114,101,109,95,115,0,4,5,114,101,109,95,117,0,5,8,103,101,116,95,104,105,103,104,0,0,10,191,1,6,4,0,35,0,11,36,1,1,126,32,0,173,32,1,173,66,32,134,132,32,2,173,32,3,173,66,32,134,132,126,34,4,66,32,135,167,36,0,32,4,167,11,36,1,1,126,32,0,173,32,1,173,66,32,134,132,32,2,173,32,3,173,66,32,134,132,127,34,4,66,32,135,167,36,0,32,4,167,11,36,1,1,126,32,0,173,32,1,173,66,32,134,132,32,2,173,32,3,173,66,32,134,132,128,34,4,66,32,135,167,36,0,32,4,167,11,36,1,1,126,32,0,173,32,1,173,66,32,134,132,32,2,173,32,3,173,66,32,134,132,129,34,4,66,32,135,167,36,0,32,4,167,11,36,1,1,126,32,0,173,32,1,173,66,32,134,132,32,2,173,32,3,173,66,32,134,132,130,34,4,66,32,135,167,36,0,32,4,167,11])),{}).exports}catch{}function G(e,t,n){this.low=e|0,this.high=t|0,this.unsigned=!!n}G.prototype.__isLong__;Object.defineProperty(G.prototype,"__isLong__",{value:!0});function pt(e){return(e&&e.__isLong__)===!0}G.isLong=pt;var Ls={},Us={};function ae(e,t){var n,s,r;return t?(e>>>=0,(r=0<=e&&e<256)&&(s=Us[e],s)?s:(n=H(e,(e|0)<0?-1:0,!0),r&&(Us[e]=n),n)):(e|=0,(r=-128<=e&&e<128)&&(s=Ls[e],s)?s:(n=H(e,e<0?-1:0,!1),r&&(Ls[e]=n),n))}G.fromInt=ae;function $t(e,t){if(isNaN(e))return t?Xt:xt;if(t){if(e<0)return Xt;if(e>=yr)return kr}else{if(e<=-qs)return lt;if(e+1>=qs)return xr}return e<0?$t(-e,t).neg():H(e%ke|0,e/ke|0,t)}G.fromNumber=$t;function H(e,t,n){return new G(e,t,n)}G.fromBits=H;var on=Math.pow;function ss(e,t,n){if(e.length===0)throw Error("empty string");if(e==="NaN"||e==="Infinity"||e==="+Infinity"||e==="-Infinity")return xt;if(typeof t=="number"?(n=t,t=!1):t=!!t,n=n||10,n<2||36<n)throw RangeError("radix");var s;if((s=e.indexOf("-"))>0)throw Error("interior hyphen");if(s===0)return ss(e.substring(1),t,n).neg();for(var r=$t(on(n,8)),o=xt,a=0;a<e.length;a+=8){var i=Math.min(8,e.length-a),c=parseInt(e.substring(a,a+i),n);if(i<8){var u=$t(on(n,i));o=o.mul(u).add($t(c))}else o=o.mul(r),o=o.add($t(c))}return o.unsigned=t,o}G.fromString=ss;function St(e,t){return typeof e=="number"?$t(e,t):typeof e=="string"?ss(e,t):H(e.low,e.high,typeof t=="boolean"?t:e.unsigned)}G.fromValue=St;var Ws=65536,cu=1<<24,ke=Ws*Ws,yr=ke*ke,qs=yr/2,Ks=ae(cu),xt=ae(0);G.ZERO=xt;var Xt=ae(0,!0);G.UZERO=Xt;var de=ae(1);G.ONE=de;var $r=ae(1,!0);G.UONE=$r;var Bn=ae(-1);G.NEG_ONE=Bn;var xr=H(-1,2147483647,!1);G.MAX_VALUE=xr;var kr=H(-1,-1,!0);G.MAX_UNSIGNED_VALUE=kr;var lt=H(0,-2147483648,!1);G.MIN_VALUE=lt;var v=G.prototype;v.toInt=function(){return this.unsigned?this.low>>>0:this.low};v.toNumber=function(){return this.unsigned?(this.high>>>0)*ke+(this.low>>>0):this.high*ke+(this.low>>>0)};v.toString=function(t){if(t=t||10,t<2||36<t)throw RangeError("radix");if(this.isZero())return"0";if(this.isNegative())if(this.eq(lt)){var n=$t(t),s=this.div(n),r=s.mul(n).sub(this);return s.toString(t)+r.toInt().toString(t)}else return"-"+this.neg().toString(t);for(var o=$t(on(t,6),this.unsigned),a=this,i="";;){var c=a.div(o),u=a.sub(c.mul(o)).toInt()>>>0,h=u.toString(t);if(a=c,a.isZero())return h+i;for(;h.length<6;)h="0"+h;i=""+h+i}};v.getHighBits=function(){return this.high};v.getHighBitsUnsigned=function(){return this.high>>>0};v.getLowBits=function(){return this.low};v.getLowBitsUnsigned=function(){return this.low>>>0};v.getNumBitsAbs=function(){if(this.isNegative())return this.eq(lt)?64:this.neg().getNumBitsAbs();for(var t=this.high!=0?this.high:this.low,n=31;n>0&&!(t&1<<n);n--);return this.high!=0?n+33:n+1};v.isZero=function(){return this.high===0&&this.low===0};v.eqz=v.isZero;v.isNegative=function(){return!this.unsigned&&this.high<0};v.isPositive=function(){return this.unsigned||this.high>=0};v.isOdd=function(){return(this.low&1)===1};v.isEven=function(){return(this.low&1)===0};v.equals=function(t){return pt(t)||(t=St(t)),this.unsigned!==t.unsigned&&this.high>>>31===1&&t.high>>>31===1?!1:this.high===t.high&&this.low===t.low};v.eq=v.equals;v.notEquals=function(t){return!this.eq(t)};v.neq=v.notEquals;v.ne=v.notEquals;v.lessThan=function(t){return this.comp(t)<0};v.lt=v.lessThan;v.lessThanOrEqual=function(t){return this.comp(t)<=0};v.lte=v.lessThanOrEqual;v.le=v.lessThanOrEqual;v.greaterThan=function(t){return this.comp(t)>0};v.gt=v.greaterThan;v.greaterThanOrEqual=function(t){return this.comp(t)>=0};v.gte=v.greaterThanOrEqual;v.ge=v.greaterThanOrEqual;v.compare=function(t){if(pt(t)||(t=St(t)),this.eq(t))return 0;var n=this.isNegative(),s=t.isNegative();return n&&!s?-1:!n&&s?1:this.unsigned?t.high>>>0>this.high>>>0||t.high===this.high&&t.low>>>0>this.low>>>0?-1:1:this.sub(t).isNegative()?-1:1};v.comp=v.compare;v.negate=function(){return!this.unsigned&&this.eq(lt)?lt:this.not().add(de)};v.neg=v.negate;v.add=function(t){pt(t)||(t=St(t));var n=this.high>>>16,s=this.high&65535,r=this.low>>>16,o=this.low&65535,a=t.high>>>16,i=t.high&65535,c=t.low>>>16,u=t.low&65535,h=0,l=0,d=0,g=0;return g+=o+u,d+=g>>>16,g&=65535,d+=r+c,l+=d>>>16,d&=65535,l+=s+i,h+=l>>>16,l&=65535,h+=n+a,h&=65535,H(d<<16|g,h<<16|l,this.unsigned)};v.subtract=function(t){return pt(t)||(t=St(t)),this.add(t.neg())};v.sub=v.subtract;v.multiply=function(t){if(this.isZero())return xt;if(pt(t)||(t=St(t)),yt){var n=yt.mul(this.low,this.high,t.low,t.high);return H(n,yt.get_high(),this.unsigned)}if(t.isZero())return xt;if(this.eq(lt))return t.isOdd()?lt:xt;if(t.eq(lt))return this.isOdd()?lt:xt;if(this.isNegative())return t.isNegative()?this.neg().mul(t.neg()):this.neg().mul(t).neg();if(t.isNegative())return this.mul(t.neg()).neg();if(this.lt(Ks)&&t.lt(Ks))return $t(this.toNumber()*t.toNumber(),this.unsigned);var s=this.high>>>16,r=this.high&65535,o=this.low>>>16,a=this.low&65535,i=t.high>>>16,c=t.high&65535,u=t.low>>>16,h=t.low&65535,l=0,d=0,g=0,w=0;return w+=a*h,g+=w>>>16,w&=65535,g+=o*h,d+=g>>>16,g&=65535,g+=a*u,d+=g>>>16,g&=65535,d+=r*h,l+=d>>>16,d&=65535,d+=o*u,l+=d>>>16,d&=65535,d+=a*c,l+=d>>>16,d&=65535,l+=s*h+r*u+o*c+a*i,l&=65535,H(g<<16|w,l<<16|d,this.unsigned)};v.mul=v.multiply;v.divide=function(t){if(pt(t)||(t=St(t)),t.isZero())throw Error("division by zero");if(yt){if(!this.unsigned&&this.high===-2147483648&&t.low===-1&&t.high===-1)return this;var n=(this.unsigned?yt.div_u:yt.div_s)(this.low,this.high,t.low,t.high);return H(n,yt.get_high(),this.unsigned)}if(this.isZero())return this.unsigned?Xt:xt;var s,r,o;if(this.unsigned){if(t.unsigned||(t=t.toUnsigned()),t.gt(this))return Xt;if(t.gt(this.shru(1)))return $r;o=Xt}else{if(this.eq(lt)){if(t.eq(de)||t.eq(Bn))return lt;if(t.eq(lt))return de;var a=this.shr(1);return s=a.div(t).shl(1),s.eq(xt)?t.isNegative()?de:Bn:(r=this.sub(t.mul(s)),o=s.add(r.div(t)),o)}else if(t.eq(lt))return this.unsigned?Xt:xt;if(this.isNegative())return t.isNegative()?this.neg().div(t.neg()):this.neg().div(t).neg();if(t.isNegative())return this.div(t.neg()).neg();o=xt}for(r=this;r.gte(t);){s=Math.max(1,Math.floor(r.toNumber()/t.toNumber()));for(var i=Math.ceil(Math.log(s)/Math.LN2),c=i<=48?1:on(2,i-48),u=$t(s),h=u.mul(t);h.isNegative()||h.gt(r);)s-=c,u=$t(s,this.unsigned),h=u.mul(t);u.isZero()&&(u=de),o=o.add(u),r=r.sub(h)}return o};v.div=v.divide;v.modulo=function(t){if(pt(t)||(t=St(t)),yt){var n=(this.unsigned?yt.rem_u:yt.rem_s)(this.low,this.high,t.low,t.high);return H(n,yt.get_high(),this.unsigned)}return this.sub(this.div(t).mul(t))};v.mod=v.modulo;v.rem=v.modulo;v.not=function(){return H(~this.low,~this.high,this.unsigned)};v.and=function(t){return pt(t)||(t=St(t)),H(this.low&t.low,this.high&t.high,this.unsigned)};v.or=function(t){return pt(t)||(t=St(t)),H(this.low|t.low,this.high|t.high,this.unsigned)};v.xor=function(t){return pt(t)||(t=St(t)),H(this.low^t.low,this.high^t.high,this.unsigned)};v.shiftLeft=function(t){return pt(t)&&(t=t.toInt()),(t&=63)===0?this:t<32?H(this.low<<t,this.high<<t|this.low>>>32-t,this.unsigned):H(0,this.low<<t-32,this.unsigned)};v.shl=v.shiftLeft;v.shiftRight=function(t){return pt(t)&&(t=t.toInt()),(t&=63)===0?this:t<32?H(this.low>>>t|this.high<<32-t,this.high>>t,this.unsigned):H(this.high>>t-32,this.high>=0?0:-1,this.unsigned)};v.shr=v.shiftRight;v.shiftRightUnsigned=function(t){if(pt(t)&&(t=t.toInt()),t&=63,t===0)return this;var n=this.high;if(t<32){var s=this.low;return H(s>>>t|n<<32-t,n>>>t,this.unsigned)}else return t===32?H(n,0,this.unsigned):H(n>>>t-32,0,this.unsigned)};v.shru=v.shiftRightUnsigned;v.shr_u=v.shiftRightUnsigned;v.toSigned=function(){return this.unsigned?H(this.low,this.high,!1):this};v.toUnsigned=function(){return this.unsigned?this:H(this.low,this.high,!0)};v.toBytes=function(t){return t?this.toBytesLE():this.toBytesBE()};v.toBytesLE=function(){var t=this.high,n=this.low;return[n&255,n>>>8&255,n>>>16&255,n>>>24,t&255,t>>>8&255,t>>>16&255,t>>>24]};v.toBytesBE=function(){var t=this.high,n=this.low;return[t>>>24,t>>>16&255,t>>>8&255,t&255,n>>>24,n>>>16&255,n>>>8&255,n&255]};G.fromBytes=function(t,n,s){return s?G.fromBytesLE(t,n):G.fromBytesBE(t,n)};G.fromBytesLE=function(t,n){return new G(t[0]|t[1]<<8|t[2]<<16|t[3]<<24,t[4]|t[5]<<8|t[6]<<16|t[7]<<24,n)};G.fromBytesBE=function(t,n){return new G(t[4]<<24|t[5]<<16|t[6]<<8|t[7],t[0]<<24|t[1]<<16|t[2]<<8|t[3],n)};const vr=au(wr),uu=qo({__proto__:null,default:vr},[wr]);/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const jt=vr||uu;function je(e){return jt.fromString(e,!0,16)}const Er=je("c3a5c85c97cb3127"),Gt=je("b492b66fbe98f273"),ot=je("9ae16a3b2f90404f");function Cn(e){return e.xor(e.shru(47))}function Sr(e,t,n){const s=e.slice(t,t+n);return jt.fromBytes(Array.from(s),!0,!0)}function W(e,t){return Sr(e,t,8)}function zs(e,t){return Sr(e,t,4)}function et(e,t){return t===0?e:e.shru(t).or(e.shl(64-t))}function Pt(e,t,n=je("9ddfea08eb382d69")){let s=e.xor(t).mul(n);s=s.xor(s.shru(47));let r=t.xor(s).mul(n);return r=r.xor(r.shru(47)),r=r.mul(n),r}function lu(e,t,n,s,r,o){r=r.add(e),o=et(o.add(r).add(s),21);const a=r;return r=r.add(t),r=r.add(n),o=o.add(et(r,44)),[r.add(s),o.add(a)]}function Je(e,t,n,s){return lu(W(e,t),W(e,t+8),W(e,t+16),W(e,t+24),n,s)}function hu(e,t=e.length){if(t>=8){const n=ot.add(t*2),s=W(e,0).add(ot),r=W(e,t-8),o=et(r,37).mul(n).add(s),a=et(s,25).add(r).mul(n);return Pt(o,a,n)}if(t>=4){const n=ot.add(t*2),s=zs(e,0);return Pt(s.shl(3).add(t),zs(e,t-4),n)}if(t>0){const n=e[0],s=e[t>>1],r=e[t-1],o=n+(s<<8),a=t+(r<<2);return Cn(ot.mul(o).xor(Er.mul(a))).mul(ot)}return ot}function fu(e,t=e.length){const n=ot.add(t*2),s=W(e,0).mul(Gt),r=W(e,8),o=W(e,t-8).mul(n),a=W(e,t-16).mul(ot);return Pt(et(s.add(r),43).add(et(o,30)).add(a),s.add(et(r.add(ot),18)).add(o),n)}function du(e,t=e.length){const n=ot.add(t*2),s=W(e,0).mul(ot),r=W(e,8),o=W(e,t-8).mul(n),a=W(e,t-16).mul(ot),i=et(s.add(r),43).add(et(o,30)).add(a),c=Pt(i,s.add(et(r.add(ot),18)).add(o),n),u=W(e,16).mul(n),h=W(e,24),l=i.add(W(e,t-32)).mul(n),d=c.add(W(e,t-24)).mul(n);return Pt(et(u.add(h),43).add(et(l,30)).add(d),u.add(et(h.add(s),18)).add(l),n)}function pu(e,t=e.length){const n=jt.fromNumber(81,!0);if(t<=32)return t<=16?hu(e,t):fu(e,t);if(t<=64)return du(e,t);let s=n,r=n.mul(Gt).add(113),o=Cn(r.mul(ot).add(113)).mul(ot),a=[jt.UZERO,jt.UZERO],i=[jt.UZERO,jt.UZERO];s=s.mul(ot).add(W(e,0));let c=0;const u=(t-1>>6)*64,h=u+(t-1&63)-63;do s=et(s.add(r).add(a[0]).add(W(e,c+8)),37).mul(Gt),r=et(r.add(a[1]).add(W(e,c+48)),42).mul(Gt),s=s.xor(i[1]),r=r.add(a[0]).add(W(e,c+40)),o=et(o.add(i[0]),33).mul(Gt),a=Je(e,c,a[1].mul(Gt),s.add(i[0])),i=Je(e,c+32,o.add(i[1]),r.add(W(e,c+16))),[o,s]=[s,o],c+=64;while(c!==u);const l=Gt.add(o.and(255).shl(1));return c=h,i[0]=i[0].add(t-1&63),a[0]=a[0].add(i[0]),i[0]=i[0].add(a[0]),s=et(s.add(r).add(a[0]).add(W(e,c+8)),37).mul(l),r=et(r.add(a[1]).add(W(e,c+48)),42).mul(l),s=s.xor(i[1].mul(9)),r=r.add(a[0].mul(9).add(W(e,c+40))),o=et(o.add(i[0]),33).mul(l),a=Je(e,c,a[1].mul(l),s.add(i[0])),i=Je(e,c+32,o.add(i[1]),r.add(W(e,c+16))),[o,s]=[s,o],Pt(Pt(a[0],i[0],l).add(Cn(r).mul(Er)).add(o),Pt(a[1],i[1],l).add(s),l)}/**
 * @license
 * Copyright 2017 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function gu(e,t){return t==="string"?rs(e):bn([e],t)}function mu(e,t){return e instanceof Float32Array&&t==="float32"||e instanceof Int32Array&&t==="int32"||e instanceof Uint8Array&&t==="bool"}function bn(e,t){if(t==="string")throw new Error("Cannot convert a string[] to a TypedArray");if(Array.isArray(e)&&(e=ve(e)),B().getBool("DEBUG")&&or(e,t),mu(e,t))return e;if(t==null||t==="float32"||t==="complex64")return new Float32Array(e);if(t==="int32")return new Int32Array(e);if(t==="bool"){const n=new Uint8Array(e.length);for(let s=0;s<n.length;++s)Math.round(e[s])!==0&&(n[s]=1);return n}else throw new Error(`Unknown data type ${t}`)}function Ce(){return B().platform.now()}function bu(e,t){return B().platform.fetch(e,t)}function rs(e,t="utf-8"){return t=t||"utf-8",B().platform.encode(e,t)}function Rn(e,t="utf-8"){return t=t||"utf-8",B().platform.decode(e,t)}function bt(e){return B().platform.isTypedArray!=null?B().platform.isTypedArray(e):br(e)}function ve(e,t=[],n=!1){if(t==null&&(t=[]),typeof e=="boolean"||typeof e=="number"||typeof e=="string"||mn(e)||e==null||bt(e)&&n)t.push(e);else if(Array.isArray(e)||bt(e))for(let s=0;s<e.length;++s)ve(e[s],t,n);else{let s=-1;for(const r of Object.keys(e))/^([1-9]+[0-9]*|0)$/.test(r)&&(s=Math.max(s,Number(r)));for(let r=0;r<=s;r++)ve(e[r],t,n)}return t}const S1=Object.freeze(Object.defineProperty({__proto__:null,arraysEqual:Wt,arraysEqualWithNull:nr,assert:p,assertNonNegativeIntegerDimensions:dt,assertNonNull:re,assertShapesMatch:at,bytesFromStringArray:ir,bytesPerElement:rn,checkConversionForErrors:or,clamp:Vo,computeStrides:Ve,convertBackendValuesAndArrayBuffer:aa,createScalarValue:gu,createShuffledIndices:ea,decodeString:Rn,distSquared:Zo,encodeString:rs,fetch:bu,fingerPrint64:pu,flatten:ve,getArrayFromDType:Qn,getTypedArrayFromDType:rr,hasEncodingLoss:ra,hexToLong:je,indexToLoc:ua,inferDtype:He,inferFromImplicitShape:sa,isBoolean:cr,isFunction:Lt,isInt:$e,isNumber:ur,isPromise:mn,isScalarShape:Jo,isString:pn,isTypedArray:bt,isValidDtype:ar,locToIndex:ca,makeOnesTypedArray:ts,makeZerosNestedTypedArray:ia,makeZerosTypedArray:gn,nearestDivisor:oa,nearestLargerEven:jo,now:Ce,parseAxisParam:Ge,randUniform:Yo,repeatedTry:na,rightPad:Ae,shuffle:er,shuffleCombo:Ho,sizeFromShape:Z,sizeToSquarishShape:ta,squeezeShape:sr,sum:Xo,swap:sn,tanh:Qo,toNestedArray:pe,toTypedArray:bn},Symbol.toStringTag,{value:"Module"}));/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class wu{constructor(t,n){this.backendTimer=t,this.logger=n,n==null&&(this.logger=new $u)}profileKernel(t,n,s){let r;const o=()=>{r=s()};let a;const i=Ce();if(this.backendTimer.timerAvailable())a=this.backendTimer.time(o);else{o();for(const u of r)u.dataSync();a=Promise.resolve({kernelMs:Ce()-i})}if(B().getBool("CHECK_COMPUTATION_FOR_ERRORS"))for(let u=0;u<r.length;u++){const h=r[u];h.data().then(l=>{yu(l,h.dtype,t)})}return{kernelName:t,outputs:r,inputs:n,timeMs:a.then(u=>u.kernelMs),extraInfo:a.then(u=>u.getExtraProfileInfo!=null?u.getExtraProfileInfo():"")}}logKernelProfile(t){const{kernelName:n,outputs:s,timeMs:r,inputs:o,extraInfo:a}=t;s.forEach(i=>{Promise.all([i.data(),r,a]).then(c=>{this.logger.logKernelProfile(n,i,c[0],c[1],o,c[2])})})}}function yu(e,t,n){if(t!=="float32")return!1;for(let s=0;s<e.length;s++){const r=e[s];if(isNaN(r)||!isFinite(r))return console.warn(`Found ${r} in the result of '${n}'`),!0}return!1}class $u{logKernelProfile(t,n,s,r,o,a){const i=typeof r=="number"?Ae(`${r}ms`,9):r.error,c=Ae(t,25),u=n.rank,h=n.size,l=Ae(n.shape.toString(),14);let d="";for(const g in o){const w=o[g];if(w!=null){const y=w.shape||n.shape,$=y.length;d+=`${g}: ${$}D ${$>0?y:""} `}}console.log(`%c${c}	%c${i}	%c${u}D ${l}	%c${h}	%c${d}	%c${a}`,"font-weight:bold","color:red","color:blue","color: orange","color: green","color: steelblue")}}/**
 * @license
 * Copyright 2017 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function xu(e,t,n){const s={},r={};for(let c=0;c<t.length;c++)s[t[c].id]=!0;for(let c=0;c<e.length;c++){const u=e[c],h=u.inputs;for(const l in h){const d=h[l];let g=!1;for(let w=0;w<t.length;w++)if(s[d.id]){u.outputs.forEach(y=>s[y.id]=!0),g=!0,r[u.id]=!0;break}if(g)break}}const o={};o[n.id]=!0;const a={};for(let c=e.length-1;c>=0;c--){const u=e[c],h=u.inputs;for(let l=0;l<u.outputs.length;l++)if(o[u.outputs[l].id]){for(const d in h)o[h[d].id]=!0,a[u.id]=!0;break}}const i=[];for(let c=0;c<e.length;c++){const u=e[c];if(r[u.id]&&a[u.id]){const h={};for(const d in u.inputs){const g=u.inputs[d];s[g.id]&&(h[d]=g)}const l=Object.assign({},u);l.inputs=h,l.outputs=u.outputs,i.push(l)}}return i}function ku(e,t,n,s){for(let r=t.length-1;r>=0;r--){const o=t[r],a=[];if(o.outputs.forEach(c=>{const u=e[c.id];u!=null?a.push(u):a.push(null)}),o.gradient==null)throw new Error(`Cannot compute gradient: gradient function not found for ${o.kernelName}.`);const i=o.gradient(a);for(const c in o.inputs){if(!(c in i))throw new Error(`Cannot backprop through input ${c}. Available gradients found: ${Object.keys(i)}.`);const u=n(()=>i[c]());if(u.dtype!=="float32")throw new Error(`Error in gradient for op ${o.kernelName}. The gradient of input ${c} must have 'float32' dtype, but has '${u.dtype}'`);const h=o.inputs[c];if(!Wt(u.shape,h.shape))throw new Error(`Error in gradient for op ${o.kernelName}. The gradient of input '${c}' has shape '${u.shape}', which does not match the shape of the input '${h.shape}'`);if(e[h.id]==null)e[h.id]=u;else{const l=e[h.id];e[h.id]=s(l,u),l.dispose()}}}}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Gs=20,Ie=3,Dn=7;function vu(e,t,n,s){const r=Ve(t),o=Eu(e,t,n,r),a=t.length,i=tn(e,t,n,r,o),c=["Tensor"];return s&&(c.push(`  dtype: ${n}`),c.push(`  rank: ${a}`),c.push(`  shape: [${t}]`),c.push("  values:")),c.push(i.map(u=>"    "+u).join(`
`)),c.join(`
`)}function Eu(e,t,n,s){const r=Z(t),o=s[s.length-1],a=new Array(o).fill(0),i=t.length,c=n==="complex64"?Ne(e):e;if(i>1)for(let u=0;u<r/o;u++){const h=u*o;for(let l=0;l<o;l++)a[l]=Math.max(a[l],De(c[h+l],0,n).length)}return a}function De(e,t,n){let s;return Array.isArray(e)?s=`${parseFloat(e[0].toFixed(Dn))} + ${parseFloat(e[1].toFixed(Dn))}j`:pn(e)?s=`'${e}'`:n==="bool"?s=Tr(e):s=parseFloat(e.toFixed(Dn)).toString(),Ae(s,t)}function Tr(e){return e===0?"false":"true"}function tn(e,t,n,s,r,o=!0){const a=n==="complex64"?2:1,i=t[0],c=t.length;if(c===0){if(n==="complex64"){const y=Ne(e);return[De(y[0],0,n)]}return n==="bool"?[Tr(e[0])]:[e[0].toString()]}if(c===1){if(i>Gs){const $=Ie*a;let k=Array.from(e.slice(0,$)),I=Array.from(e.slice((i-Ie)*a,i*a));return n==="complex64"&&(k=Ne(k),I=Ne(I)),["["+k.map((D,E)=>De(D,r[E],n)).join(", ")+", ..., "+I.map((D,E)=>De(D,r[i-Ie+E],n)).join(", ")+"]"]}return["["+(n==="complex64"?Ne(e):Array.from(e)).map(($,k)=>De($,r[k],n)).join(", ")+"]"]}const u=t.slice(1),h=s.slice(1),l=s[0]*a,d=[];if(i>Gs){for(let y=0;y<Ie;y++){const $=y*l,k=$+l;d.push(...tn(e.slice($,k),u,n,h,r,!1))}d.push("...");for(let y=i-Ie;y<i;y++){const $=y*l,k=$+l;d.push(...tn(e.slice($,k),u,n,h,r,y===i-1))}}else for(let y=0;y<i;y++){const $=y*l,k=$+l;d.push(...tn(e.slice($,k),u,n,h,r,y===i-1))}const g=c===2?",":"";d[0]="["+(i>0?d[0]+g:"");for(let y=1;y<d.length-1;y++)d[y]=" "+d[y]+g;let w=`,
`;for(let y=2;y<c;y++)w+=`
`;return d[d.length-1]=" "+d[d.length-1]+"]"+(o?"":w),d}function Ne(e){const t=[];for(let n=0;n<e.length;n+=2)t.push([e[n],e[n+1]]);return t}/**
 * @license
 * Copyright 2017 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class Pn{constructor(t,n,s){if(this.dtype=n,this.shape=t.slice(),this.size=Z(t),s!=null){const r=s.length;p(r===this.size,()=>`Length of values '${r}' does not match the size inferred by the shape '${this.size}'.`)}if(n==="complex64")throw new Error("complex64 dtype TensorBuffers are not supported. Please create a TensorBuffer for the real and imaginary parts separately and call tf.complex(real, imag).");this.values=s||Qn(n,this.size),this.strides=Ve(t)}set(t,...n){n.length===0&&(n=[0]),p(n.length===this.rank,()=>`The number of provided coordinates (${n.length}) must match the rank (${this.rank})`);const s=this.locToIndex(n);this.values[s]=t}get(...t){t.length===0&&(t=[0]);let n=0;for(const r of t){if(r<0||r>=this.shape[n]){const o=`Requested out of range element at ${t}.   Buffer shape=${this.shape}`;throw new Error(o)}n++}let s=t[t.length-1];for(let r=0;r<t.length-1;++r)s+=this.strides[r]*t[r];return this.values[s]}locToIndex(t){if(this.rank===0)return 0;if(this.rank===1)return t[0];let n=t[t.length-1];for(let s=0;s<t.length-1;++s)n+=this.strides[s]*t[s];return n}indexToLoc(t){if(this.rank===0)return[];if(this.rank===1)return[t];const n=new Array(this.shape.length);for(let s=0;s<n.length-1;++s)n[s]=Math.floor(t/this.strides[s]),t-=n[s]*this.strides[s];return n[n.length-1]=t,n}get rank(){return this.shape.length}toTensor(){return kt().makeTensor(this.values,this.shape,this.dtype)}}let kt=null,he=null;function Su(e){kt=e}function Tu(e){he=e}class nt{constructor(t,n,s,r){this.kept=!1,this.isDisposedInternal=!1,this.shape=t.slice(),this.dtype=n||"float32",this.size=Z(t),this.strides=Ve(t),this.dataId=s,this.id=r,this.rankType=this.rank<5?this.rank.toString():"higher"}get rank(){return this.shape.length}async buffer(){const t=await this.data();return he.buffer(this.shape,this.dtype,t)}bufferSync(){return he.buffer(this.shape,this.dtype,this.dataSync())}async array(){const t=await this.data();return pe(this.shape,t,this.dtype==="complex64")}arraySync(){return pe(this.shape,this.dataSync(),this.dtype==="complex64")}async data(){this.throwIfDisposed();const t=kt().read(this.dataId);if(this.dtype==="string"){const n=await t;try{return n.map(s=>Rn(s))}catch{throw new Error("Failed to decode the string bytes into utf-8. To get the original bytes, call tensor.bytes().")}}return t}dataToGPU(t){return this.throwIfDisposed(),kt().readToGPU(this.dataId,t)}dataSync(){this.throwIfDisposed();const t=kt().readSync(this.dataId);if(this.dtype==="string")try{return t.map(n=>Rn(n))}catch{throw new Error("Failed to decode the string bytes into utf-8. To get the original bytes, call tensor.bytes().")}return t}async bytes(){this.throwIfDisposed();const t=await kt().read(this.dataId);return this.dtype==="string"?t:new Uint8Array(t.buffer)}dispose(){this.isDisposed||(kt().disposeTensor(this),this.isDisposedInternal=!0)}get isDisposed(){return this.isDisposedInternal}throwIfDisposed(){if(this.isDisposed)throw new Error("Tensor is disposed.")}print(t=!1){return he.print(this,t)}clone(){return this.throwIfDisposed(),he.clone(this)}toString(t=!1){const n=this.dataSync();return vu(n,this.shape,this.dtype,t)}cast(t){return this.throwIfDisposed(),he.cast(this,t)}variable(t=!0,n,s){return this.throwIfDisposed(),kt().makeVariable(this,t,n,s)}}Object.defineProperty(nt,Symbol.hasInstance,{value:e=>!!e&&e.data!=null&&e.dataSync!=null&&e.throwIfDisposed!=null});function _u(){return es("Tensor",()=>nt)}_u();class an extends nt{constructor(t,n,s,r){super(t.shape,t.dtype,t.dataId,r),this.trainable=n,this.name=s}assign(t){if(t.dtype!==this.dtype)throw new Error(`dtype of the new value (${t.dtype}) and previous value (${this.dtype}) must match`);if(!Wt(t.shape,this.shape))throw new Error(`shape of the new value (${t.shape}) and previous value (${this.shape}) must match`);kt().disposeTensor(this),this.dataId=t.dataId,kt().incRef(this,null)}dispose(){kt().disposeVariable(this),this.isDisposedInternal=!0}}Object.defineProperty(an,Symbol.hasInstance,{value:e=>e instanceof nt&&e.assign!=null&&e.assign instanceof Function});/**
 * @license
 * Copyright 2017 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */var Hs;(function(e){e.R0="R0",e.R1="R1",e.R2="R2",e.R3="R3",e.R4="R4",e.R5="R5",e.R6="R6"})(Hs||(Hs={}));var On;(function(e){e.float32="float32",e.int32="int32",e.bool="int32",e.complex64="complex64"})(On||(On={}));var Ln;(function(e){e.float32="float32",e.int32="int32",e.bool="bool",e.complex64="complex64"})(Ln||(Ln={}));var Un;(function(e){e.float32="float32",e.int32="float32",e.bool="float32",e.complex64="complex64"})(Un||(Un={}));var Wn;(function(e){e.float32="complex64",e.int32="complex64",e.bool="complex64",e.complex64="complex64"})(Wn||(Wn={}));const Iu={float32:Un,int32:On,bool:Ln,complex64:Wn};function _r(e,t){if(e==="string"||t==="string"){if(e==="string"&&t==="string")return"string";throw new Error(`Can not upcast ${e} with ${t}`)}return Iu[e][t]}function T1(e){return _r(e,"int32")}function Ir(e){return e!=null&&typeof e=="object"&&"texture"in e&&e.texture instanceof WebGLTexture}function Dr(e){return typeof GPUBuffer<"u"&&e!=null&&typeof e=="object"&&"buffer"in e&&e.buffer instanceof GPUBuffer}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Y(e,t){if(e.dtype===t.dtype)return[e,t];const n=_r(e.dtype,t.dtype);return[e.cast(n),t.cast(n)]}function Nr(e,t){p(e.dtype===t.dtype,()=>`The dtypes of the first(${e.dtype}) and second(${t.dtype}) input must match`)}function Du(e,t){return t.some(n=>n.id===e.id)}function os(e){const t=[];return Ar(e,t,new Set),t}function Ar(e,t,n){if(e==null)return;if(e instanceof nt){t.push(e);return}if(!Nu(e))return;const s=e;for(const r in s){const o=s[r];n.has(o)||(n.add(o),Ar(o,t,n))}}function Nu(e){return Array.isArray(e)||typeof e=="object"}const _1=Object.freeze(Object.defineProperty({__proto__:null,assertTypesMatch:Nr,getTensorsInContainer:os,isTensorInList:Du,makeTypesMatch:Y},Symbol.toStringTag,{value:"Module"}));/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Nn(e){return e.kernelName!=null}class Vs{constructor(){this.registeredVariables={},this.nextTapeNodeId=0,this.numBytes=0,this.numTensors=0,this.numStringTensors=0,this.numDataBuffers=0,this.gradientDepth=0,this.kernelDepth=0,this.scopeStack=[],this.numDataMovesStack=[],this.nextScopeId=0,this.tensorInfo=new WeakMap,this.profiling=!1,this.activeProfile={newBytes:0,newTensors:0,peakBytes:0,kernels:[],result:null,get kernelNames(){return Array.from(new Set(this.kernels.map(t=>t.name)))}}}dispose(){for(const t in this.registeredVariables)this.registeredVariables[t].dispose()}}class Ee{constructor(t){this.ENV=t,this.registry={},this.registryFactory={},this.pendingBackendInitId=0,this.state=new Vs}async ready(){if(this.pendingBackendInit!=null)return this.pendingBackendInit.then(()=>{});if(this.backendInstance!=null)return;const t=this.getSortedBackends();for(let n=0;n<t.length;n++){const s=t[n];if(await this.initializeBackend(s).success){await this.setBackend(s);return}}throw new Error("Could not initialize any backends, all backend initializations failed.")}get backend(){if(this.pendingBackendInit!=null)throw new Error(`Backend '${this.backendName}' has not yet been initialized. Make sure to await tf.ready() or await tf.setBackend() before calling other methods`);if(this.backendInstance==null){const{name:t,asyncInit:n}=this.initializeBackendsAndReturnBest();if(n)throw new Error(`The highest priority backend '${t}' has not yet been initialized. Make sure to await tf.ready() or await tf.setBackend() before calling other methods`);this.setBackend(t)}return this.backendInstance}backendNames(){return Object.keys(this.registryFactory)}findBackend(t){if(!(t in this.registry))if(t in this.registryFactory){const{asyncInit:n}=this.initializeBackend(t);if(n)return null}else return null;return this.registry[t]}findBackendFactory(t){return t in this.registryFactory?this.registryFactory[t].factory:null}registerBackend(t,n,s=1){return t in this.registryFactory?(Vt(`${t} backend was already registered. Reusing existing backend factory.`),!1):(this.registryFactory[t]={factory:n,priority:s},!0)}async setBackend(t){if(this.registryFactory[t]==null)throw new Error(`Backend name '${t}' not found in registry`);if(this.backendName=t,this.registry[t]==null){this.backendInstance=null;const{success:n,asyncInit:s}=this.initializeBackend(t);if(!(s?await n:n))return!1}return this.backendInstance=this.registry[t],this.setupRegisteredKernels(),this.profiler=new wu(this.backendInstance),!0}setupRegisteredKernels(){Fn(this.backendName).forEach(n=>{n.setupFunc!=null&&n.setupFunc(this.backendInstance)})}disposeRegisteredKernels(t){Fn(t).forEach(s=>{s.disposeFunc!=null&&s.disposeFunc(this.registry[t])})}initializeBackend(t){const n=this.registryFactory[t];if(n==null)throw new Error(`Cannot initialize backend ${t}, no registration found.`);try{const s=n.factory();if(s&&!(s instanceof Go)&&typeof s.then=="function"){const r=++this.pendingBackendInitId,o=s.then(a=>r<this.pendingBackendInitId?!1:(this.registry[t]=a,this.pendingBackendInit=null,!0)).catch(a=>(r<this.pendingBackendInitId||(this.pendingBackendInit=null,Vt(`Initialization of backend ${t} failed`),Vt(a.stack||a.message)),!1));return this.pendingBackendInit=o,{success:o,asyncInit:!0}}else return this.registry[t]=s,{success:!0,asyncInit:!1}}catch(s){return Vt(`Initialization of backend ${t} failed`),Vt(s.stack||s.message),{success:!1,asyncInit:!1}}}removeBackend(t){if(!(t in this.registryFactory))throw new Error(`${t} backend not found in registry`);this.backendName===t&&this.pendingBackendInit!=null&&this.pendingBackendInitId++,t in this.registry&&(this.disposeRegisteredKernels(t),this.registry[t].dispose(),delete this.registry[t]),delete this.registryFactory[t],this.backendName===t&&(this.pendingBackendInit=null,this.backendName=null,this.backendInstance=null)}getSortedBackends(){if(Object.keys(this.registryFactory).length===0)throw new Error("No backend found in registry.");return Object.keys(this.registryFactory).sort((t,n)=>this.registryFactory[n].priority-this.registryFactory[t].priority)}initializeBackendsAndReturnBest(){const t=this.getSortedBackends();for(let n=0;n<t.length;n++){const s=t[n],{success:r,asyncInit:o}=this.initializeBackend(s);if(o||r)return{name:s,asyncInit:o}}throw new Error("Could not initialize any backends, all backend initializations failed.")}moveData(t,n){const s=this.state.tensorInfo.get(n),r=s.backend,o=this.readSync(n),a=r.refCount(n);r.disposeData(n,!0),s.backend=t,t.move(n,o,s.shape,s.dtype,a),this.shouldCheckForMemLeaks()&&this.state.numDataMovesStack[this.state.numDataMovesStack.length-1]++}tidy(t,n){let s=null;if(n==null){if(typeof t!="function")throw new Error("Please provide a function to tidy()");n=t}else{if(typeof t!="string"&&!(t instanceof String))throw new Error("When calling with two arguments, the first argument to tidy() must be a string");if(typeof n!="function")throw new Error("When calling with two arguments, the 2nd argument to tidy() must be a function");s=t}let r;return this.scopedRun(()=>this.startScope(s),()=>this.endScope(r),()=>(r=n(),r instanceof Promise&&console.error("Cannot return a Promise inside of tidy."),r))}scopedRun(t,n,s){t();try{const r=s();return n(),r}catch(r){throw n(),r}}nextTensorId(){return Ee.nextTensorId++}nextVariableId(){return Ee.nextVariableId++}clone(t){const n=b.runKernel(gr,{x:t}),s={x:t},r=a=>({x:()=>{const i="float32",c={x:a},u={dtype:i};return b.runKernel(pr,c,u)}}),o=[];return this.addTapeNode(this.state.activeScope.name,s,[n],r,o,{}),n}runKernel(t,n,s){if(this.backendName==null&&this.backend,!(Ps(t,this.backendName)!=null))throw new Error(`Kernel '${t}' not registered for backend '${this.backendName}'`);return this.runKernelFunc({kernelName:t,inputs:n,attrs:s})}shouldCheckForMemLeaks(){return this.ENV.getBool("IS_TEST")}checkKernelForMemLeak(t,n,s){const r=this.backend.numDataIds();let o=0;s.forEach(c=>{o+=c.dtype==="complex64"?3:1});const a=this.state.numDataMovesStack[this.state.numDataMovesStack.length-1],i=r-n-o-a;if(i>0)throw new Error(`Backend '${this.backendName}' has an internal memory leak (${i} data ids) after running '${t}'`)}runKernelFunc(t){let n,s=[];const r=this.isTapeOn(),o=this.state.numBytes,a=this.state.numTensors;this.shouldCheckForMemLeaks()&&this.state.numDataMovesStack.push(0);let i;this.backendName==null&&this.backend;let c;const u=Nn(t)?t.kernelName:this.state.activeScope!=null?this.state.activeScope.name:"";if(Nn(t)){const{kernelName:w,inputs:y,attrs:$}=t;this.backendName==null&&this.backend;const k=Ps(w,this.backendName);p(k!=null,()=>`Cannot find registered kernel '${w}' for backend '${this.backendName}'`),i=()=>{const I=this.backend.numDataIds();c=k.kernelFunc({inputs:y,attrs:$,backend:this.backend});const D=Array.isArray(c)?c:[c];this.shouldCheckForMemLeaks()&&this.checkKernelForMemLeak(w,I,D);const E=D.map(T=>T.rank!=null?T:this.makeTensorFromTensorInfo(T));if(r){const T=this.getTensorsForGradient(w,y,E);s=this.saveTensorsForBackwardMode(T)}return E}}else{const{forwardFunc:w}=t,y=$=>{r&&(s=$.map(k=>this.keep(this.clone(k))))};i=()=>{const $=this.backend.numDataIds();c=this.tidy(()=>w(this.backend,y));const k=Array.isArray(c)?c:[c];return this.shouldCheckForMemLeaks()&&this.checkKernelForMemLeak(u,$,k),k}}const{inputs:h,attrs:l}=t,d=Nn(t)?null:t.backwardsFunc;let g;return this.scopedRun(()=>this.state.kernelDepth++,()=>this.state.kernelDepth--,()=>{!this.ENV.getBool("DEBUG")&&!this.state.profiling?n=i():(g=this.profiler.profileKernel(u,h,()=>i()),this.ENV.getBool("DEBUG")&&this.profiler.logKernelProfile(g),n=g.outputs)}),r&&this.addTapeNode(u,h,n,d,s,l),this.state.profiling&&this.state.activeProfile.kernels.push({name:u,bytesAdded:this.state.numBytes-o,totalBytesSnapshot:this.state.numBytes,tensorsAdded:this.state.numTensors-a,totalTensorsSnapshot:this.state.numTensors,inputShapes:Object.keys(h).map(w=>h[w]!=null?h[w].shape:null),outputShapes:n.map(w=>w.shape),kernelTimeMs:g.timeMs,extraInfo:g.extraInfo}),Array.isArray(c)?n:n[0]}saveTensorsForBackwardMode(t){return t.map(s=>this.keep(this.clone(s)))}getTensorsForGradient(t,n,s){const r=Os(t);if(r!=null){const o=r.inputsToSave||[],a=r.outputsToSave||[];let i;r.saveAllInputs?(p(Array.isArray(n),()=>"saveAllInputs is true, expected inputs to be an array."),i=Object.keys(n).map(u=>n[u])):i=o.map(u=>n[u]);const c=s.filter((u,h)=>a[h]);return i.concat(c)}return[]}makeTensor(t,n,s,r){if(t==null)throw new Error("Values passed to engine.makeTensor() are null");s=s||"float32",r=r||this.backend;let o=t;s==="string"&&pn(t[0])&&(o=t.map(c=>rs(c)));const a=r.write(o,n,s),i=new nt(n,s,a,this.nextTensorId());if(this.trackTensor(i,r),s==="string"){const c=this.state.tensorInfo.get(a),u=ir(o);this.state.numBytes+=u-c.bytes,c.bytes=u}return i}makeTensorFromDataId(t,n,s,r){s=s||"float32";const o={dataId:t,shape:n,dtype:s};return this.makeTensorFromTensorInfo(o,r)}makeTensorFromTensorInfo(t,n){const{dataId:s,shape:r,dtype:o}=t,a=new nt(r,o,s,this.nextTensorId());return this.trackTensor(a,n),a}makeVariable(t,n=!0,s,r){s=s||this.nextVariableId().toString(),r!=null&&r!==t.dtype&&(t=t.cast(r));const o=new an(t,n,s,this.nextTensorId());if(this.state.registeredVariables[o.name]!=null)throw new Error(`Variable with name ${o.name} was already registered`);return this.state.registeredVariables[o.name]=o,this.incRef(o,this.backend),o}trackTensor(t,n){this.state.numTensors++,t.dtype==="string"&&this.state.numStringTensors++;let s=0;t.dtype!=="complex64"&&t.dtype!=="string"&&(s=t.size*rn(t.dtype)),this.state.numBytes+=s,this.state.tensorInfo.has(t.dataId)||(this.state.numDataBuffers++,this.state.tensorInfo.set(t.dataId,{backend:n||this.backend,dtype:t.dtype,shape:t.shape,bytes:s})),t instanceof an||this.track(t)}incRef(t,n){this.trackTensor(t,n),this.backend.incRef(t.dataId)}removeDataId(t,n){this.state.tensorInfo.has(t)&&this.state.tensorInfo.get(t).backend===n&&(this.state.tensorInfo.delete(t),this.state.numDataBuffers--)}disposeTensor(t){if(!this.state.tensorInfo.has(t.dataId))return;const n=this.state.tensorInfo.get(t.dataId);if(this.state.numTensors--,t.dtype==="string"&&(this.state.numStringTensors--,this.state.numBytes-=n.bytes),t.dtype!=="complex64"&&t.dtype!=="string"){const s=t.size*rn(t.dtype);this.state.numBytes-=s}n.backend.disposeData(t.dataId)&&this.removeDataId(t.dataId,n.backend)}disposeVariables(){for(const t in this.state.registeredVariables){const n=this.state.registeredVariables[t];this.disposeVariable(n)}}disposeVariable(t){this.disposeTensor(t),this.state.registeredVariables[t.name]!=null&&delete this.state.registeredVariables[t.name]}memory(){const t=this.backend.memory();return t.numTensors=this.state.numTensors,t.numDataBuffers=this.state.numDataBuffers,t.numBytes=this.state.numBytes,this.state.numStringTensors>0&&(t.unreliable=!0,t.reasons==null&&(t.reasons=[]),t.reasons.push("Memory usage by string tensors is approximate (2 bytes per character)")),t}async profile(t){this.state.profiling=!0;const n=this.state.numBytes,s=this.state.numTensors;this.state.activeProfile.kernels=[],this.state.activeProfile.result=await t(),this.state.profiling=!1,this.state.activeProfile.peakBytes=Math.max(...this.state.activeProfile.kernels.map(r=>r.totalBytesSnapshot)),this.state.activeProfile.newBytes=this.state.numBytes-n,this.state.activeProfile.newTensors=this.state.numTensors-s;for(const r of this.state.activeProfile.kernels)r.kernelTimeMs=await r.kernelTimeMs,r.extraInfo=await r.extraInfo;return this.state.activeProfile}isTapeOn(){return this.state.gradientDepth>0&&this.state.kernelDepth===0}addTapeNode(t,n,s,r,o,a){const i={id:this.state.nextTapeNodeId++,kernelName:t,inputs:n,outputs:s,saved:o},c=Os(t);c!=null&&(r=c.gradFunc),r!=null&&(i.gradient=u=>(u=u.map((h,l)=>{if(h==null){const d=s[l],g=gn(d.size,d.dtype);return this.makeTensor(g,d.shape,d.dtype)}return h}),r(u.length>1?u:u[0],o,a))),this.state.activeTape.push(i)}keep(t){return t.kept=!0,t}startTape(){this.state.gradientDepth===0&&(this.state.activeTape=[]),this.state.gradientDepth++}endTape(){this.state.gradientDepth--}startScope(t){const n={track:[],name:"unnamed scope",id:this.state.nextScopeId++};t&&(n.name=t),this.state.scopeStack.push(n),this.state.activeScope=n}endScope(t){const n=os(t),s=new Set(n.map(o=>o.id));for(let o=0;o<this.state.activeScope.track.length;o++){const a=this.state.activeScope.track[o];!a.kept&&!s.has(a.id)&&a.dispose()}const r=this.state.scopeStack.pop();this.state.activeScope=this.state.scopeStack.length===0?null:this.state.scopeStack[this.state.scopeStack.length-1],n.forEach(o=>{!o.kept&&o.scopeId===r.id&&this.track(o)})}gradients(t,n,s,r=!1){if(p(n.length>0,()=>"gradients() received an empty list of xs."),s!=null&&s.dtype!=="float32")throw new Error(`dy must have 'float32' dtype, but has '${s.dtype}'`);const o=this.scopedRun(()=>this.startTape(),()=>this.endTape(),()=>this.tidy("forward",t));p(o instanceof nt,()=>"The result y returned by f() must be a tensor.");const a=xu(this.state.activeTape,n,o);if(!r&&a.length===0&&n.length>0)throw new Error("Cannot compute gradient of y=f(x) with respect to x. Make sure that the f you passed encloses all operations that lead from x to y.");return this.tidy("backward",()=>{const i={};i[o.id]=s??Au(o.shape),ku(i,a,u=>this.tidy(u),Mu);const c=n.map(u=>i[u.id]);return this.state.gradientDepth===0&&(this.state.activeTape.forEach(u=>{for(const h of u.saved)h.dispose()}),this.state.activeTape=null),{value:o,grads:c}})}customGrad(t){return p(Lt(t),()=>"The f passed in customGrad(f) must be a function."),(...n)=>{p(n.every(i=>i instanceof nt),()=>"The args passed in customGrad(f)(x1, x2,...) must all be tensors");let s;const r={};n.forEach((i,c)=>{r[c]=i});const o=(i,c)=>(s=t(...n,c),p(s.value instanceof nt,()=>"The function f passed in customGrad(f) must return an object where `obj.value` is a tensor"),p(Lt(s.gradFunc),()=>"The function f passed in customGrad(f) must return an object where `obj.gradFunc` is a function."),s.value),a=(i,c)=>{const u=s.gradFunc(i,c),h=Array.isArray(u)?u:[u];p(h.length===n.length,()=>"The function f passed in customGrad(f) must return an object where `obj.gradFunc` is a function that returns the same number of tensors as inputs passed to f(...)."),p(h.every(d=>d instanceof nt),()=>"The function f passed in customGrad(f) must return an object where `obj.gradFunc` is a function that returns a list of only tensors.");const l={};return h.forEach((d,g)=>{l[g]=()=>d}),l};return this.runKernelFunc({forwardFunc:o,backwardsFunc:a,inputs:r})}}readSync(t){return this.state.tensorInfo.get(t).backend.readSync(t)}read(t){return this.state.tensorInfo.get(t).backend.read(t)}readToGPU(t,n){return this.state.tensorInfo.get(t).backend.readToGPU(t,n)}async time(t){const n=Ce(),s=await this.backend.time(t);return s.wallMs=Ce()-n,s}track(t){return this.state.activeScope!=null&&(t.scopeId=this.state.activeScope.id,this.state.activeScope.track.push(t)),t}get registeredVariables(){return this.state.registeredVariables}reset(){this.pendingBackendInitId++,this.state.dispose(),this.ENV.reset(),this.state=new Vs;for(const t in this.registry)this.disposeRegisteredKernels(t),this.registry[t].dispose(),delete this.registry[t];this.backendName=null,this.backendInstance=null,this.pendingBackendInit=null}}Ee.nextTensorId=0;Ee.nextVariableId=0;function Au(e){const t=ts(Z(e),"float32");return b.makeTensor(t,e,"float32")}function Mr(){const e=fr();if(e._tfengine==null){const t=new la(e);e._tfengine=new Ee(t)}return pa(e._tfengine.ENV),Su(()=>e._tfengine),e._tfengine}const b=Mr();function Mu(e,t){const n={a:e,b:t};return b.runKernel(dr,n)}/**
 * @license
 * Copyright 2017 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Fu(){return typeof navigator<"u"&&navigator!=null}let qn;function Bu(e){qn=e}function Cu(e){if(qn!==void 0)return qn;if(e||Fu()){if(e||(e=navigator),e.product==="ReactNative")return!0;const t=e.userAgent||e.vendor||(typeof window<"u"?window.opera:"");if(!t){const n=e;return n.userAgentData&&n.userAgentData.mobile}return/(android|bb\d+|meego).+mobile|avantgo|bada\/|blackberry|blazer|compal|elaine|fennec|hiptop|iemobile|ip(hone|od)|iris|kindle|lge |maemo|midp|mmp|mobile.+firefox|netfront|opera m(ob|in)i|palm( os)?|phone|p(ixi|re)\/|plucker|pocket|psp|series(4|6)0|symbian|treo|up\.(browser|link)|vodafone|wap|windows ce|xda|xiino/i.test(t)||/1207|6310|6590|3gso|4thp|50[1-6]i|770s|802s|a wa|abac|ac(er|oo|s\-)|ai(ko|rn)|al(av|ca|co)|amoi|an(ex|ny|yw)|aptu|ar(ch|go)|as(te|us)|attw|au(di|\-m|r |s )|avan|be(ck|ll|nq)|bi(lb|rd)|bl(ac|az)|br(e|v)w|bumb|bw\-(n|u)|c55\/|capi|ccwa|cdm\-|cell|chtm|cldc|cmd\-|co(mp|nd)|craw|da(it|ll|ng)|dbte|dc\-s|devi|dica|dmob|do(c|p)o|ds(12|\-d)|el(49|ai)|em(l2|ul)|er(ic|k0)|esl8|ez([4-7]0|os|wa|ze)|fetc|fly(\-|_)|g1 u|g560|gene|gf\-5|g\-mo|go(\.w|od)|gr(ad|un)|haie|hcit|hd\-(m|p|t)|hei\-|hi(pt|ta)|hp( i|ip)|hs\-c|ht(c(\-| |_|a|g|p|s|t)|tp)|hu(aw|tc)|i\-(20|go|ma)|i230|iac( |\-|\/)|ibro|idea|ig01|ikom|im1k|inno|ipaq|iris|ja(t|v)a|jbro|jemu|jigs|kddi|keji|kgt( |\/)|klon|kpt |kwc\-|kyo(c|k)|le(no|xi)|lg( g|\/(k|l|u)|50|54|\-[a-w])|libw|lynx|m1\-w|m3ga|m50\/|ma(te|ui|xo)|mc(01|21|ca)|m\-cr|me(rc|ri)|mi(o8|oa|ts)|mmef|mo(01|02|bi|de|do|t(\-| |o|v)|zz)|mt(50|p1|v )|mwbp|mywa|n10[0-2]|n20[2-3]|n30(0|2)|n50(0|2|5)|n7(0(0|1)|10)|ne((c|m)\-|on|tf|wf|wg|wt)|nok(6|i)|nzph|o2im|op(ti|wv)|oran|owg1|p800|pan(a|d|t)|pdxg|pg(13|\-([1-8]|c))|phil|pire|pl(ay|uc)|pn\-2|po(ck|rt|se)|prox|psio|pt\-g|qa\-a|qc(07|12|21|32|60|\-[2-7]|i\-)|qtek|r380|r600|raks|rim9|ro(ve|zo)|s55\/|sa(ge|ma|mm|ms|ny|va)|sc(01|h\-|oo|p\-)|sdk\/|se(c(\-|0|1)|47|mc|nd|ri)|sgh\-|shar|sie(\-|m)|sk\-0|sl(45|id)|sm(al|ar|b3|it|t5)|so(ft|ny)|sp(01|h\-|v\-|v )|sy(01|mb)|t2(18|50)|t6(00|10|18)|ta(gt|lk)|tcl\-|tdg\-|tel(i|m)|tim\-|t\-mo|to(pl|sh)|ts(70|m\-|m3|m5)|tx\-9|up(\.b|g1|si)|utst|v400|v750|veri|vi(rg|te)|vk(40|5[0-3]|\-v)|vm40|voda|vulc|vx(52|53|60|61|70|80|81|83|85|98)|w3c(\-| )|webc|whit|wi(g |nc|nw)|wmlb|wonu|x700|yas\-|your|zeto|zte\-/i.test(t.substr(0,4))}return!1}function Fr(){return typeof window<"u"&&window.document!=null||typeof WorkerGlobalScope<"u"}const I1=Object.freeze(Object.defineProperty({__proto__:null,isBrowser:Fr,isMobile:Cu,mockIsMobile:Bu},Symbol.toStringTag,{value:"Module"}));/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const ct=B();ct.registerFlag("DEBUG",()=>!1,e=>{e&&console.warn("Debugging mode is ON. The output of every math call will be downloaded to CPU and checked for NaNs. This significantly impacts performance.")});ct.registerFlag("IS_BROWSER",()=>Fr());ct.registerFlag("IS_NODE",()=>typeof process<"u"&&typeof process.versions<"u"&&typeof process.versions.node<"u");ct.registerFlag("IS_CHROME",()=>typeof navigator<"u"&&navigator!=null&&navigator.userAgent!=null&&/Chrome/.test(navigator.userAgent)&&/Google Inc/.test(navigator.vendor));ct.registerFlag("IS_SAFARI",()=>typeof navigator<"u"&&navigator!=null&&navigator.userAgent!=null&&/Safari/.test(navigator.userAgent)&&/Apple/.test(navigator.vendor));ct.registerFlag("PROD",()=>!1);ct.registerFlag("TENSORLIKE_CHECK_SHAPE_CONSISTENCY",()=>ct.getBool("DEBUG"));ct.registerFlag("DEPRECATION_WARNINGS_ENABLED",()=>!0);ct.registerFlag("IS_TEST",()=>!1);ct.registerFlag("CHECK_COMPUTATION_FOR_ERRORS",()=>ct.getBool("DEBUG"));ct.registerFlag("WRAP_TO_IMAGEBITMAP",()=>!1);ct.registerFlag("CANVAS2D_WILL_READ_FREQUENTLY_FOR_GPU",()=>!1);ct.registerFlag("USE_SETTIMEOUTCUSTOM",()=>!1);/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function qt(e,t){let n=e;if(bt(e))return t==="string"?[]:[e.length];if(Ir(e)){const r=e.channels||"RGBA";return[e.height,e.width*r.length]}else if(Dr(e))return[e.buffer.size/(t==null?4:rn(t))];if(!Array.isArray(e))return[];const s=[];for(;Array.isArray(n)||bt(n)&&t!=="string";)s.push(n.length),n=n[0];return Array.isArray(e)&&B().getBool("TENSORLIKE_CHECK_SHAPE_CONSISTENCY")&&Br(e,s,[]),s}function Br(e,t,n){if(n=n||[],!Array.isArray(e)&&!bt(e)){p(t.length===0,()=>`Element arr[${n.join("][")}] is a primitive, but should be an array/TypedArray of ${t[0]} elements`);return}p(t.length>0,()=>`Element arr[${n.join("][")}] should be a primitive, but is an array of ${e.length} elements`),p(e.length===t[0],()=>`Element arr[${n.join("][")}] should have ${t[0]} elements, but has ${e.length} elements`);const s=t.slice(1);for(let r=0;r<e.length;++r)Br(e[r],s,n.concat(r))}function js(e,t,n,s){if(e!=="string_or_numeric"){if(e==null)throw new Error("Expected dtype cannot be null.");if(e!=="numeric"&&e!==t||e==="numeric"&&t==="string")throw new Error(`Argument '${n}' passed to '${s}' must be ${e} tensor, but got ${t} tensor`)}}function f(e,t,n,s="numeric"){if(e instanceof nt)return js(s,e.dtype,t,n),e;let r=He(e);if(r!=="string"&&["bool","int32","float32"].indexOf(s)>=0&&(r=s),js(s,r,t,n),e==null||!bt(e)&&!Array.isArray(e)&&typeof e!="number"&&typeof e!="boolean"&&typeof e!="string"){const c=e==null?"null":e.constructor.name;throw new Error(`Argument '${t}' passed to '${n}' must be a Tensor or TensorLike, but got '${c}'`)}const o=qt(e,r);!bt(e)&&!Array.isArray(e)&&(e=[e]);const i=r!=="string"?bn(e,r):ve(e,[],!0);return b.makeTensor(i,o,r)}function Re(e,t,n,s="numeric"){if(!Array.isArray(e))throw new Error(`Argument ${t} passed to ${n} must be a \`Tensor[]\` or \`TensorLike[]\``);return e.map((o,a)=>f(o,`${t}[${a}]`,n,s))}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Cr="__op";function m(e){const t=Object.keys(e);if(t.length!==1)throw new Error(`Please provide an object with a single key (operation name) mapping to a function. Got an object with ${t.length} keys.`);let n=t[0];const s=e[n];n.endsWith("_")&&(n=n.substring(0,n.length-1)),n=n+Cr;const r=(...o)=>{b.startScope(n);try{const a=s(...o);return mn(a)&&console.error("Cannot return a Promise inside of tidy."),b.endScope(a),a}catch(a){throw b.endScope(null),a}};return Object.defineProperty(r,"name",{value:n,configurable:!0}),r}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Ru(e,t){const n=f(e,"real","complex"),s=f(t,"imag","complex");at(n.shape,s.shape,`real and imag shapes, ${n.shape} and ${s.shape}, must match in call to tf.complex().`);const r={real:n,imag:s};return b.runKernel(Oa,r)}const Ut=m({complex_:Ru});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Kt(e,t,n,s){if(s==null)s=He(e);else if(s==="complex64")throw new Error("Cannot construct a complex64 tensor directly. Please use tf.complex(real, imag).");if(Dr(e)||Ir(e)){if(s!=="float32"&&s!=="int32")throw new Error(`Creating tensor from GPU data only supports 'float32'|'int32' dtype, while the dtype is ${s}.`);return b.backend.createTensorFromGPUData(e,t||n,s)}if(!bt(e)&&!Array.isArray(e)&&typeof e!="number"&&typeof e!="boolean"&&typeof e!="string")throw new Error("values passed to tensor(values) must be a number/boolean/string or an array of numbers/booleans/strings, or a TypedArray");if(t!=null){dt(t);const r=Z(t),o=Z(n);p(r===o,()=>`Based on the provided shape, [${t}], the tensor should have ${r} values but has ${o}`);for(let a=0;a<n.length;++a){const i=n[a],c=a===n.length-1?i!==Z(t.slice(a)):!0;p(n[a]===t[a]||!c,()=>`Error creating a new Tensor. Inferred shape (${n}) does not match the provided shape (${t}). `)}}return!bt(e)&&!Array.isArray(e)&&(e=[e]),t=t||n,e=s!=="string"?bn(e,s):ve(e,[],!0),b.makeTensor(e,t,s)}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function ge(e,t,n){const s=qt(e,n);return Kt(e,t,s,n)}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Kn={float32:4,float16:2,int32:4,uint16:2,uint8:1,bool:1,complex64:8};class It{static join(t){return new It(t).slice()}constructor(t){if(this.shards=[],this.previousShardIndex=0,t==null||(t instanceof Array||(t=[t]),t=t.map(s=>bt(s)?s.buffer:s),t.length===0))return;this.bufferUniformSize=t[0].byteLength;let n=0;for(let s=0;s<t.length;s++){const r=t[s];s!==t.length-1&&r.byteLength!==this.bufferUniformSize&&(this.bufferUniformSize=void 0);const o=n+r.byteLength;this.shards.push({buffer:r,start:n,end:o}),n=o}this.shards.length===0&&(this.byteLength=0),this.byteLength=this.shards[this.shards.length-1].end}slice(t=0,n=this.byteLength){if(this.shards.length===0)return new ArrayBuffer(0);if(t=isNaN(Number(t))?0:t,n=isNaN(Number(n))?0:n,t=Math.max(0,t),n=Math.min(this.byteLength,n),n<=t)return new ArrayBuffer(0);const s=this.findShardForByte(t);if(s===-1)throw new Error(`Could not find start shard for byte ${t}`);const r=n-t,o=new ArrayBuffer(r),a=new Uint8Array(o);let i=0;for(let c=s;c<this.shards.length;c++){const u=this.shards[c],l=t+i-u.start,d=i,w=Math.min(n,u.end)-u.start,y=new Uint8Array(u.buffer,l,w-l);if(a.set(y,d),i+=y.length,n<u.end)break}return o}findShardForByte(t){if(this.shards.length===0||t<0||t>=this.byteLength)return-1;if(this.bufferUniformSize!=null)return this.previousShardIndex=Math.floor(t/this.bufferUniformSize),this.previousShardIndex;function n(r){return t<r.start?-1:t>=r.end?1:0}if(n(this.shards[this.previousShardIndex])===0)return this.previousShardIndex;const s=Pu(this.shards,n);return s===-1?-1:(this.previousShardIndex=s,this.previousShardIndex)}}function Pu(e,t){let n=0,s=e.length;for(;n<=s;){const r=Math.floor((s-n)/2)+n,o=t(e[r]);if(o===0)return r;o<0?s=r:n=r+1}return-1}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const cn=4;async function Ou(e,t){const n=[],s=[],r=Array.isArray(e)?e.map(a=>a.name):Object.keys(e);for(let a=0;a<r.length;++a){const i=r[a],c=Array.isArray(e)?e[a].tensor:e[i];if(c.dtype!=="float32"&&c.dtype!=="int32"&&c.dtype!=="bool"&&c.dtype!=="string"&&c.dtype!=="complex64")throw new Error(`Unsupported dtype in weight '${i}': ${c.dtype}`);const u={name:i,shape:c.shape,dtype:c.dtype};if(c.dtype==="string"){const h=new Promise(async l=>{const d=await c.bytes(),g=d.reduce(($,k)=>$+k.length,0)+cn*d.length,w=new Uint8Array(g);let y=0;for(let $=0;$<d.length;$++){const k=d[$],I=new Uint8Array(new Uint32Array([k.length]).buffer);w.set(I,y),y+=cn,w.set(k,y),y+=k.length}l(w)});s.push(h)}else s.push(c.data());t!=null&&(u.group=t),n.push(u)}const o=await Promise.all(s);return{data:Lu(o),specs:n}}function Rr(e,t){const n=new It(e),s={};let r,o=0;for(const a of t){const i=a.name,c=a.dtype,u=a.shape,h=Z(u);let l;if("quantization"in a){const d=a.quantization;if(d.dtype==="uint8"||d.dtype==="uint16"){if(!("min"in d&&"scale"in d))throw new Error(`Weight ${a.name} with quantization ${d.dtype} doesn't have corresponding metadata min and scale.`)}else if(d.dtype==="float16"){if(c!=="float32")throw new Error(`Weight ${a.name} is quantized with ${d.dtype} which only supports weights of type float32 not ${c}.`)}else throw new Error(`Weight ${a.name} has unknown quantization dtype ${d.dtype}. Supported quantization dtypes are: 'uint8', 'uint16', and 'float16'.`);const g=Kn[d.dtype],w=n.slice(o,o+h*g),y=d.dtype==="uint8"?new Uint8Array(w):new Uint16Array(w);if(c==="float32")if(d.dtype==="uint8"||d.dtype==="uint16"){l=new Float32Array(y.length);for(let $=0;$<y.length;$++){const k=y[$];l[$]=k*d.scale+d.min}}else if(d.dtype==="float16")r===void 0&&(r=Hu()),l=r(y);else throw new Error(`Unsupported quantization type ${d.dtype} for weight type float32.`);else if(c==="int32"){if(d.dtype!=="uint8"&&d.dtype!=="uint16")throw new Error(`Unsupported quantization type ${d.dtype} for weight type int32.`);l=new Int32Array(y.length);for(let $=0;$<y.length;$++){const k=y[$];l[$]=Math.round(k*d.scale+d.min)}}else throw new Error(`Unsupported dtype in weight '${i}': ${c}`);o+=h*g}else if(c==="string"){const d=Z(a.shape);l=[];for(let g=0;g<d;g++){const w=new Uint32Array(n.slice(o,o+cn))[0];o+=cn;const y=new Uint8Array(n.slice(o,o+w));l.push(y),o+=w}}else{const d=Kn[c],g=n.slice(o,o+h*d);if(c==="float32")l=new Float32Array(g);else if(c==="int32")l=new Int32Array(g);else if(c==="bool")l=new Uint8Array(g);else if(c==="complex64"){l=new Float32Array(g);const w=new Float32Array(l.length/2),y=new Float32Array(l.length/2);for(let I=0;I<w.length;I++)w[I]=l[I*2],y[I]=l[I*2+1];const $=ge(w,u,"float32"),k=ge(y,u,"float32");s[i]=Ut($,k),$.dispose(),k.dispose()}else throw new Error(`Unsupported dtype in weight '${i}': ${c}`);o+=h*d}c!=="complex64"&&(s[i]=ge(l,u,c))}return s}function Lu(e){if(e===null)throw new Error(`Invalid input value: ${JSON.stringify(e)}`);let t=0;const n=[];e.forEach(o=>{if(t+=o.byteLength,n.push(o.byteLength===o.buffer.byteLength?o:new o.constructor(o)),!(o instanceof Float32Array||o instanceof Int32Array||o instanceof Uint8Array))throw new Error(`Unsupported TypedArray subtype: ${o.constructor.name}`)});const s=new Uint8Array(t);let r=0;return n.forEach(o=>{s.set(new Uint8Array(o.buffer),r),r+=o.byteLength}),s.buffer}const as=typeof Buffer<"u"&&(typeof Blob>"u"||typeof atob>"u"||typeof btoa>"u");function Xs(e){return as?Buffer.byteLength(e,"utf8"):new Blob([e]).size}function Uu(e){if(as)return Buffer.from(e).toString("base64");const t=new Uint8Array(e);let n="";for(let s=0,r=t.length;s<r;s++)n+=String.fromCharCode(t[s]);return btoa(n)}function Wu(e){if(as){const s=Buffer.from(e,"base64");return s.buffer.slice(s.byteOffset,s.byteOffset+s.byteLength)}const t=atob(e),n=new Uint8Array(t.length);for(let s=0;s<t.length;++s)n.set([t.charCodeAt(s)],s);return n.buffer}function qu(e){return It.join(e)}function Ys(e){const t="/";for(e=e.trim();e.endsWith(t);)e=e.slice(0,e.length-1);const n=e.split(t);return n[n.length-1]}function Pr(e,t){const n={modelTopology:e.modelTopology,format:e.format,generatedBy:e.generatedBy,convertedBy:e.convertedBy,weightsManifest:t};return e.signature!=null&&(n.signature=e.signature),e.userDefinedMetadata!=null&&(n.userDefinedMetadata=e.userDefinedMetadata),e.modelInitializer!=null&&(n.modelInitializer=e.modelInitializer),e.initializerSignature!=null&&(n.initializerSignature=e.initializerSignature),e.trainingConfig!=null&&(n.trainingConfig=e.trainingConfig),n}function Or(e,t,n){const s={modelTopology:e.modelTopology,format:e.format,generatedBy:e.generatedBy,convertedBy:e.convertedBy};if(e.trainingConfig!=null&&(s.trainingConfig=e.trainingConfig),e.weightsManifest!=null){if(!t)throw new Error("modelJSON has weightsManifest but weightSpecs is null");if(!n)throw new Error("modelJSON has weightsManifest but weightData is null");s.weightSpecs=t,s.weightData=n}return e.signature!=null&&(s.signature=e.signature),e.userDefinedMetadata!=null&&(s.userDefinedMetadata=e.userDefinedMetadata),e.modelInitializer!=null&&(s.modelInitializer=e.modelInitializer),e.initializerSignature!=null&&(s.initializerSignature=e.initializerSignature),s}async function is(e,t){let n,s;return e.weightsManifest!=null&&([n,s]=await t(e.weightsManifest)),Or(e,n,s)}function Xe(e){if(e.modelTopology instanceof ArrayBuffer)throw new Error("Expected JSON model topology, received ArrayBuffer.");return{dateSaved:new Date,modelTopologyType:"JSON",modelTopologyBytes:e.modelTopology==null?0:Xs(JSON.stringify(e.modelTopology)),weightSpecsBytes:e.weightSpecs==null?0:Xs(JSON.stringify(e.weightSpecs)),weightDataBytes:e.weightData==null?0:new It(e.weightData).byteLength}}function Lr(e){const t=[];for(const n of e)t.push(...n.weights);return t}function Ku(){const e=n=>{let s=n<<13,r=0;for(;!(s&8388608);)r-=8388608,s<<=1;return s&=-8388609,r+=947912704,s|r},t=new Uint32Array(2048);t[0]=0;for(let n=1;n<1024;n++)t[n]=e(n);for(let n=1024;n<2048;n++)t[n]=939524096+(n-1024<<13);return t}function zu(){const e=new Uint32Array(64);e[0]=0,e[31]=1199570944,e[32]=2147483648,e[63]=3347054592;for(let t=1;t<31;t++)e[t]=t<<23;for(let t=33;t<63;t++)e[t]=2147483648+(t-32<<23);return e}function Gu(){const e=new Uint32Array(64);for(let t=0;t<64;t++)e[t]=1024;return e[0]=e[32]=0,e}function Hu(){const e=Ku(),t=zu(),n=Gu();return s=>{const r=new ArrayBuffer(4*s.length),o=new Uint32Array(r);for(let a=0;a<s.length;a++){const i=s[a],c=e[n[i>>10]+(i&1023)]+t[i>>10];o[a]=c}return new Float32Array(r)}}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class X{constructor(){this.saveRouters=[],this.loadRouters=[]}static getInstance(){return X.instance==null&&(X.instance=new X),X.instance}static registerSaveRouter(t){X.getInstance().saveRouters.push(t)}static registerLoadRouter(t){X.getInstance().loadRouters.push(t)}static getSaveHandlers(t){return X.getHandlers(t,"save")}static getLoadHandlers(t,n){return X.getHandlers(t,"load",n)}static getHandlers(t,n,s){const r=[];return(n==="load"?X.getInstance().loadRouters:X.getInstance().saveRouters).forEach(a=>{const i=a(t,s);i!==null&&r.push(i)}),r}}const Vu=e=>X.registerSaveRouter(e),ju=e=>X.registerLoadRouter(e),Xu=e=>X.getSaveHandlers(e),Yu=(e,t)=>X.getLoadHandlers(e,t);/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const zn="tensorflowjs",Gn=1,Yt="models_store",Rt="model_info_store";function Ur(){if(!B().getBool("IS_BROWSER"))throw new Error("Failed to obtain IndexedDB factory because the current environmentis not a web browser.");const e=typeof window>"u"?self:window,t=e.indexedDB||e.mozIndexedDB||e.webkitIndexedDB||e.msIndexedDB||e.shimIndexedDB;if(t==null)throw new Error("The current browser does not appear to support IndexedDB.");return t}function Hn(e){const t=e.result;t.createObjectStore(Yt,{keyPath:"modelPath"}),t.createObjectStore(Rt,{keyPath:"modelPath"})}class Qt{constructor(t){if(this.indexedDB=Ur(),t==null||!t)throw new Error("For IndexedDB, modelPath must not be null, undefined or empty.");this.modelPath=t}async save(t){if(t.modelTopology instanceof ArrayBuffer)throw new Error("BrowserLocalStorage.save() does not support saving model topology in binary formats yet.");return this.databaseAction(this.modelPath,t)}async load(){return this.databaseAction(this.modelPath)}databaseAction(t,n){return new Promise((s,r)=>{const o=this.indexedDB.open(zn,Gn);o.onupgradeneeded=()=>Hn(o),o.onsuccess=()=>{const a=o.result;if(n==null){const i=a.transaction(Yt,"readonly"),u=i.objectStore(Yt).get(this.modelPath);u.onsuccess=()=>{if(u.result==null)return a.close(),r(new Error(`Cannot find model with path '${this.modelPath}' in IndexedDB.`));s(u.result.modelArtifacts)},u.onerror=h=>(a.close(),r(u.error)),i.oncomplete=()=>a.close()}else{const i=Xe(n),c=a.transaction(Rt,"readwrite");let u=c.objectStore(Rt),h;try{h=u.put({modelPath:this.modelPath,modelArtifactsInfo:i})}catch(d){return r(d)}let l;h.onsuccess=()=>{l=a.transaction(Yt,"readwrite");const d=l.objectStore(Yt);let g;try{g=d.put({modelPath:this.modelPath,modelArtifacts:n,modelArtifactsInfo:i})}catch(w){return r(w)}g.onsuccess=()=>s({modelArtifactsInfo:i}),g.onerror=w=>{u=c.objectStore(Rt);const y=u.delete(this.modelPath);y.onsuccess=()=>(a.close(),r(g.error)),y.onerror=$=>(a.close(),r(g.error))}},h.onerror=d=>(a.close(),r(h.error)),c.oncomplete=()=>{l==null?a.close():l.oncomplete=()=>a.close()}}},o.onerror=a=>r(o.error)})}}Qt.URL_SCHEME="indexeddb://";const Wr=e=>B().getBool("IS_BROWSER")&&!Array.isArray(e)&&e.startsWith(Qt.URL_SCHEME)?Zu(e.slice(Qt.URL_SCHEME.length)):null;X.registerSaveRouter(Wr);X.registerLoadRouter(Wr);function Zu(e){return new Qt(e)}function Ju(e){return e.startsWith(Qt.URL_SCHEME)?e.slice(Qt.URL_SCHEME.length):e}class Qu{constructor(){this.indexedDB=Ur()}async listModels(){return new Promise((t,n)=>{const s=this.indexedDB.open(zn,Gn);s.onupgradeneeded=()=>Hn(s),s.onsuccess=()=>{const r=s.result,o=r.transaction(Rt,"readonly"),i=o.objectStore(Rt).getAll();i.onsuccess=()=>{const c={};for(const u of i.result)c[u.modelPath]=u.modelArtifactsInfo;t(c)},i.onerror=c=>(r.close(),n(i.error)),o.oncomplete=()=>r.close()},s.onerror=r=>n(s.error)})}async removeModel(t){return t=Ju(t),new Promise((n,s)=>{const r=this.indexedDB.open(zn,Gn);r.onupgradeneeded=()=>Hn(r),r.onsuccess=()=>{const o=r.result,a=o.transaction(Rt,"readwrite"),i=a.objectStore(Rt),c=i.get(t);let u;c.onsuccess=()=>{if(c.result==null)return o.close(),s(new Error(`Cannot find model with path '${t}' in IndexedDB.`));{const h=i.delete(t),l=()=>{u=o.transaction(Yt,"readwrite");const g=u.objectStore(Yt).delete(t);g.onsuccess=()=>n(c.result.modelArtifactsInfo),g.onerror=w=>s(c.error)};h.onsuccess=l,h.onerror=d=>(l(),o.close(),s(c.error))}},c.onerror=h=>(o.close(),s(c.error)),a.oncomplete=()=>{u==null?o.close():u.oncomplete=()=>o.close()}},r.onerror=o=>s(r.error)})}}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Nt="/",fe="tensorflowjs_models",qr="info",tl="model_topology",el="weight_specs",nl="weight_data",sl="model_metadata";function Kr(e){return{info:[fe,e,qr].join(Nt),topology:[fe,e,tl].join(Nt),weightSpecs:[fe,e,el].join(Nt),weightData:[fe,e,nl].join(Nt),modelMetadata:[fe,e,sl].join(Nt)}}function zr(e){for(const t of Object.values(e))window.localStorage.removeItem(t)}function rl(e){const t=e.split(Nt);if(t.length<3)throw new Error(`Invalid key format: ${e}`);return t.slice(1,t.length-1).join(Nt)}function ol(e){return e.startsWith(te.URL_SCHEME)?e.slice(te.URL_SCHEME.length):e}class te{constructor(t){if(!B().getBool("IS_BROWSER")||typeof window>"u"||typeof window.localStorage>"u")throw new Error("The current environment does not support local storage.");if(this.LS=window.localStorage,t==null||!t)throw new Error("For local storage, modelPath must not be null, undefined or empty.");this.modelPath=t,this.keys=Kr(this.modelPath)}async save(t){if(t.modelTopology instanceof ArrayBuffer)throw new Error("BrowserLocalStorage.save() does not support saving model topology in binary formats yet.");{const n=JSON.stringify(t.modelTopology),s=JSON.stringify(t.weightSpecs),r=Xe(t),o=It.join(t.weightData);try{this.LS.setItem(this.keys.info,JSON.stringify(r)),this.LS.setItem(this.keys.topology,n),this.LS.setItem(this.keys.weightSpecs,s),this.LS.setItem(this.keys.weightData,Uu(o));const a={format:t.format,generatedBy:t.generatedBy,convertedBy:t.convertedBy,signature:t.signature!=null?t.signature:void 0,userDefinedMetadata:t.userDefinedMetadata!=null?t.userDefinedMetadata:void 0,modelInitializer:t.modelInitializer!=null?t.modelInitializer:void 0,initializerSignature:t.initializerSignature!=null?t.initializerSignature:void 0,trainingConfig:t.trainingConfig!=null?t.trainingConfig:void 0};return this.LS.setItem(this.keys.modelMetadata,JSON.stringify(a)),{modelArtifactsInfo:r}}catch{throw zr(this.keys),new Error(`Failed to save model '${this.modelPath}' to local storage: size quota being exceeded is a possible cause of this failure: modelTopologyBytes=${r.modelTopologyBytes}, weightSpecsBytes=${r.weightSpecsBytes}, weightDataBytes=${r.weightDataBytes}.`)}}}async load(){const t=JSON.parse(this.LS.getItem(this.keys.info));if(t==null)throw new Error(`In local storage, there is no model with name '${this.modelPath}'`);if(t.modelTopologyType!=="JSON")throw new Error("BrowserLocalStorage does not support loading non-JSON model topology yet.");const n={},s=JSON.parse(this.LS.getItem(this.keys.topology));if(s==null)throw new Error(`In local storage, the topology of model '${this.modelPath}' is missing.`);n.modelTopology=s;const r=JSON.parse(this.LS.getItem(this.keys.weightSpecs));if(r==null)throw new Error(`In local storage, the weight specs of model '${this.modelPath}' are missing.`);n.weightSpecs=r;const o=this.LS.getItem(this.keys.modelMetadata);if(o!=null){const i=JSON.parse(o);n.format=i.format,n.generatedBy=i.generatedBy,n.convertedBy=i.convertedBy,i.signature!=null&&(n.signature=i.signature),i.userDefinedMetadata!=null&&(n.userDefinedMetadata=i.userDefinedMetadata),i.modelInitializer!=null&&(n.modelInitializer=i.modelInitializer),i.initializerSignature!=null&&(n.initializerSignature=i.initializerSignature),i.trainingConfig!=null&&(n.trainingConfig=i.trainingConfig)}const a=this.LS.getItem(this.keys.weightData);if(a==null)throw new Error(`In local storage, the binary weight values of model '${this.modelPath}' are missing.`);return n.weightData=Wu(a),n}}te.URL_SCHEME="localstorage://";const Gr=e=>B().getBool("IS_BROWSER")&&!Array.isArray(e)&&e.startsWith(te.URL_SCHEME)?al(e.slice(te.URL_SCHEME.length)):null;X.registerSaveRouter(Gr);X.registerLoadRouter(Gr);function al(e){return new te(e)}class il{constructor(){p(B().getBool("IS_BROWSER"),()=>"Current environment is not a web browser"),p(typeof window>"u"||typeof window.localStorage<"u",()=>"Current browser does not appear to support localStorage"),this.LS=window.localStorage}async listModels(){const t={},n=fe+Nt,s=Nt+qr;for(let r=0;r<this.LS.length;++r){const o=this.LS.key(r);if(o.startsWith(n)&&o.endsWith(s)){const a=rl(o);t[a]=JSON.parse(this.LS.getItem(o))}}return t}async removeModel(t){t=ol(t);const n=Kr(t);if(this.LS.getItem(n.info)==null)throw new Error(`Cannot find model at path '${t}'`);const s=JSON.parse(this.LS.getItem(n.info));return zr(n),s}}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const me="://";class rt{constructor(){this.managers={}}static getInstance(){return rt.instance==null&&(rt.instance=new rt),rt.instance}static registerManager(t,n){p(t!=null,()=>"scheme must not be undefined or null."),t.endsWith(me)&&(t=t.slice(0,t.indexOf(me))),p(t.length>0,()=>"scheme must not be an empty string.");const s=rt.getInstance();p(s.managers[t]==null,()=>`A model store manager is already registered for scheme '${t}'.`),s.managers[t]=n}static getManager(t){const n=rt.getInstance().managers[t];if(n==null)throw new Error(`Cannot find model manager for scheme '${t}'`);return n}static getSchemes(){return Object.keys(rt.getInstance().managers)}}function en(e){if(e.indexOf(me)===-1)throw new Error(`The url string provided does not contain a scheme. Supported schemes are: ${rt.getSchemes().join(",")}`);return{scheme:e.split(me)[0],path:e.split(me)[1]}}async function Hr(e,t,n=!1){p(e!==t,()=>`Old path and new path are the same: '${e}'`);const s=X.getLoadHandlers(e);p(s.length>0,()=>`Copying failed because no load handler is found for source URL ${e}.`),p(s.length<2,()=>`Copying failed because more than one (${s.length}) load handlers for source URL ${e}.`);const r=s[0],o=X.getSaveHandlers(t);p(o.length>0,()=>`Copying failed because no save handler is found for destination URL ${t}.`),p(o.length<2,()=>`Copying failed because more than one (${s.length}) save handlers for destination URL ${t}.`);const a=o[0],i=en(e).scheme,c=en(e).path,u=i===en(e).scheme,h=await r.load();n&&u&&await rt.getManager(i).removeModel(c);const l=await a.save(h);return n&&!u&&await rt.getManager(i).removeModel(c),l.modelArtifactsInfo}async function cl(){const e=rt.getSchemes(),t={};for(const n of e){const s=await rt.getManager(n).listModels();for(const r in s){const o=n+me+r;t[o]=s[r]}}return t}async function ul(e){const t=en(e);return rt.getManager(t.scheme).removeModel(t.path)}async function ll(e,t){return Hr(e,t,!1)}async function hl(e,t){return Hr(e,t,!0)}/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class fl{constructor(){this.messageName="setTimeoutCustom",this.functionRefs=[],this.handledMessageCount=0,this.hasEventListener=!1}fetch(t,n){return fetch(t,n)}now(){return performance.now()}encode(t,n){if(n!=="utf-8"&&n!=="utf8")throw new Error(`Browser's encoder only supports utf-8, but got ${n}`);return this.textEncoder==null&&(this.textEncoder=new TextEncoder),this.textEncoder.encode(t)}decode(t,n){return new TextDecoder(n).decode(t)}setTimeoutCustom(t,n){if(typeof window>"u"||!B().getBool("USE_SETTIMEOUTCUSTOM")){setTimeout(t,n);return}this.functionRefs.push(t),setTimeout(()=>{window.postMessage({name:this.messageName,index:this.functionRefs.length-1},"*")},n),this.hasEventListener||(this.hasEventListener=!0,window.addEventListener("message",s=>{if(s.source===window&&s.data.name===this.messageName){s.stopPropagation();const r=this.functionRefs[s.data.index];r(),this.handledMessageCount++,this.handledMessageCount===this.functionRefs.length&&(this.functionRefs=[],this.handledMessageCount=0)}},!0))}isTypedArray(t){return br(t)}}if(B().get("IS_BROWSER")){B().setPlatform("browser",new fl);try{rt.registerManager(te.URL_SCHEME,new il)}catch{}try{rt.registerManager(Qt.URL_SCHEME,new Qu)}catch{}}/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const dl={importFetch:()=>require("node-fetch")};let An;class pl{constructor(){this.util=require("util"),this.textEncoder=new this.util.TextEncoder}fetch(t,n){return B().global.fetch!=null?B().global.fetch(t,n):(An==null&&(An=dl.importFetch()),An(t,n))}now(){const t=process.hrtime();return t[0]*1e3+t[1]/1e6}encode(t,n){if(n!=="utf-8"&&n!=="utf8")throw new Error(`Node built-in encoder only supports utf-8, but got ${n}`);return this.textEncoder.encode(t)}decode(t,n){return t.length===0?"":new this.util.TextDecoder(n).decode(t)}isTypedArray(t){return this.util.types.isFloat32Array(t)||this.util.types.isInt32Array(t)||this.util.types.isUint8Array(t)||this.util.types.isUint8ClampedArray(t)}}B().get("IS_NODE")&&!B().get("IS_BROWSER")&&B().setPlatform("node",new pl);/**
 * @license
 * Copyright 2020 Google Inc. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function At(e,t="float32",n){return t=t||"float32",dt(e),new Pn(e,t,n)}/**
 * @license
 * Copyright 2020 Google Inc. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function gl(e,t){const n=f(e,"x","cast");if(!ar(t))throw new Error(`Failed to cast to unknown dtype ${t}`);if(t==="string"&&n.dtype!=="string"||t!=="string"&&n.dtype==="string")throw new Error("Only strings can be casted to strings");const s={x:n},r={dtype:t};return b.runKernel(pr,s,r)}const st=m({cast_:gl});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function ml(e){const n={x:f(e,"x","clone","string_or_numeric")};return b.runKernel(gr,n)}const Jt=m({clone_:ml});/**
 * @license
 * Copyright 2020 Google Inc. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Vr(e,t=!1){console.log(e.toString(t))}/**
 * @license
 * Copyright 2020 Google Inc. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */Mr();const bl={buffer:At,cast:st,clone:Jt,print:Vr};Tu(bl);/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function D1(){B().set("PROD",!0)}function N1(){B().set("DEBUG",!0)}function A1(){B().set("DEPRECATION_WARNINGS_ENABLED",!1),console.warn("TensorFlow.js deprecation warnings have been disabled.")}function M1(e){B().getBool("DEPRECATION_WARNINGS_ENABLED")&&console.warn(e+" You can disable deprecation warnings with tf.disableDeprecationWarnings().")}function F1(){b.disposeVariables()}function B1(){return b}function C1(){return b.memory()}function R1(e){return b.profile(e)}function J(e,t){return b.tidy(e,t)}function ht(e){os(e).forEach(n=>n.dispose())}function wl(e){return b.keep(e)}function P1(e){return b.time(e)}function O1(e){return b.setBackend(e)}function L1(){return b.ready()}function U1(){return b.backendName}function W1(e){b.removeBackend(e)}function q1(e){return b.findBackend(e)}function K1(e){return b.findBackendFactory(e)}function z1(e,t,n=1){return b.registerBackend(e,t,n)}function G1(){return b.backend}function H1(e,t){B().setPlatform(e,t)}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function yl(e,t){let n=f(e,"a","add"),s=f(t,"b","add");[n,s]=Y(n,s);const r={a:n,b:s};return b.runKernel(dr,r)}const M=m({add_:yl});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function $l(e,t){let n=f(e,"a","floorDiv"),s=f(t,"b","floorDiv");[n,s]=Y(n,s);const r={a:n,b:s};return b.runKernel(mi,r)}const jr=m({floorDiv_:$l});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function xl(e,t){let n=f(e,"a","div"),s=f(t,"b","div");if([n,s]=Y(n,s),n.dtype==="int32"&&s.dtype==="int32")return jr(n,s);const r={a:n,b:s},o={};return b.runKernel(ri,r,o)}const z=m({div_:xl});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function kl(e,t){let n=f(e,"a","mul"),s=f(t,"b","mul");[n,s]=Y(n,s);const r={a:n,b:s};return b.runKernel(ji,r)}const S=m({mul_:kl});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function vl(e){const t=f(e,"x","abs");if(t.dtype==="complex64"){const n={x:t};return b.runKernel(La,n)}else{const n={x:t};return b.runKernel(ma,n)}}const gt=m({abs_:vl});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function El(e){const n={x:f(e,"x","acos")};return b.runKernel(ba,n)}const Sl=m({acos_:El});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Tl(e){const n={x:f(e,"x","acosh")};return b.runKernel(wa,n)}const _l=m({acosh_:Tl});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Il(e){p(Array.isArray(e),()=>"The argument passed to tf.addN() must be a list of tensors"),p(e.length>=1,()=>`Must pass at least one tensor to tf.addN(), but got ${e.length}`);const t=e.map((r,o)=>f(r,`tensors${o}`,"addN")),n=t[0];t.forEach(r=>{if(r.dtype!==n.dtype)throw new Error("All tensors passed to tf.addN() must have the same dtype")}),t.forEach(r=>{if(!Wt(r.shape,n.shape))throw new Error("All tensors passed to tf.addN() must have the same shape")});const s=t;return b.runKernel(ya,s)}const Dl=m({addN_:Il});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Nl(e,t=null,n=!1){const r={x:f(e,"x","all","bool")},o={axis:t,keepDims:n};return b.runKernel($a,r,o)}const Al=m({all_:Nl});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Ml(e,t=null,n=!1){const r={x:f(e,"x","any","bool")},o={axis:t,keepDims:n};return b.runKernel(xa,r,o)}const Fl=m({any_:Ml});/**
 * @license
 * Copyright 2020 Google Inc. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Bl(e,t=0){const s={x:f(e,"x","argMax")},r={axis:t};return b.runKernel(ka,s,r)}const Cl=m({argMax_:Bl});/**
 * @license
 * Copyright 2020 Google Inc. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Rl(e,t=0){const s={x:f(e,"x","argMin")},r={axis:t};return b.runKernel(va,s,r)}const Pl=m({argMin_:Rl});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Ol(e){const n={x:f(e,"x","asin")};return b.runKernel(Ea,n)}const Ll=m({asin_:Ol});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Ul(e){const n={x:f(e,"x","asinh")};return b.runKernel(Sa,n)}const Wl=m({asinh_:Ul});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function ql(e){const n={x:f(e,"x","atan")};return b.runKernel(Ta,n)}const Kl=m({atan_:ql});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function zl(e,t){let n=f(e,"a","atan2"),s=f(t,"b","atan2");[n,s]=Y(n,s);const r={a:n,b:s};return b.runKernel(Ia,r)}const Gl=m({atan2_:zl});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Hl(e){const n={x:f(e,"x","atanh")};return b.runKernel(_a,n)}const Vl=m({atanh_:Hl});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function V1(e,t,n,s,r="NHWC",o){const a=e[3],i=[...t,a],c=th(r);return wn(e,i,n,o,s,null,null,c)}function jl(e,t,n,s,r,o,a="channelsLast"){const[i,c]=Pe(t);let u;if(a==="channelsLast")u=[i,c,e[3],e[3]];else if(a==="channelsFirst")u=[i,c,e[1],e[1]];else throw new Error(`Unknown dataFormat ${a}`);return wn(e,u,n,s,r,o,!1,a)}function j1(e,t,n,s,r,o,a="NDHWC"){const[i,c,u]=Vn(t);let h,l;if(a==="NDHWC")l="channelsLast",h=[i,c,u,e[4],e[4]];else if(a==="NCDHW")l="channelsFirst",h=[i,c,u,e[1],e[1]];else throw new Error(`Unknown dataFormat ${a}`);return Xl(e,h,n,s,r,!1,l,o)}function wn(e,t,n,s,r,o,a=!1,i="channelsLast"){let[c,u,h,l]=[-1,-1,-1,-1];if(i==="channelsLast")[c,u,h,l]=e;else if(i==="channelsFirst")[c,l,u,h]=e;else throw new Error(`Unknown dataFormat ${i}`);const[d,g,,w]=t,[y,$]=Pe(n),[k,I]=Pe(s),D=be(d,k),E=be(g,I),{padInfo:T,outHeight:_,outWidth:A}=Jl(r,u,h,y,$,D,E,o,i),N=a?w*l:w;let F;return i==="channelsFirst"?F=[c,N,_,A]:i==="channelsLast"&&(F=[c,_,A,N]),{batchSize:c,dataFormat:i,inHeight:u,inWidth:h,inChannels:l,outHeight:_,outWidth:A,outChannels:N,padInfo:T,strideHeight:y,strideWidth:$,filterHeight:d,filterWidth:g,effectiveFilterHeight:D,effectiveFilterWidth:E,dilationHeight:k,dilationWidth:I,inShape:e,outShape:F,filterShape:t}}function Xl(e,t,n,s,r,o=!1,a="channelsLast",i){let[c,u,h,l,d]=[-1,-1,-1,-1,-1];if(a==="channelsLast")[c,u,h,l,d]=e;else if(a==="channelsFirst")[c,d,u,h,l]=e;else throw new Error(`Unknown dataFormat ${a}`);const[g,w,y,,$]=t,[k,I,D]=Vn(n),[E,T,_]=Vn(s),A=be(g,E),N=be(w,T),F=be(y,_),{padInfo:P,outDepth:C,outHeight:q,outWidth:L}=Ql(r,u,h,l,k,I,D,A,N,F,i),V=o?$*d:$;let tt;return a==="channelsFirst"?tt=[c,V,C,q,L]:a==="channelsLast"&&(tt=[c,C,q,L,V]),{batchSize:c,dataFormat:a,inDepth:u,inHeight:h,inWidth:l,inChannels:d,outDepth:C,outHeight:q,outWidth:L,outChannels:V,padInfo:P,strideDepth:k,strideHeight:I,strideWidth:D,filterDepth:g,filterHeight:w,filterWidth:y,effectiveFilterDepth:A,effectiveFilterHeight:N,effectiveFilterWidth:F,dilationDepth:E,dilationHeight:T,dilationWidth:_,inShape:e,outShape:tt,filterShape:t}}function Yl(e,t,n,s,r){s==null&&(s=Xr(e,t,n));const o=e[0],a=e[1],i=Oe((o-t+2*s)/n+1,r),c=Oe((a-t+2*s)/n+1,r);return[i,c]}function Zl(e,t,n,s,r,o){r==null&&(r=Xr(e,t[0],s[0]));const a=[0,0,0,n];for(let i=0;i<3;i++)e[i]+2*r>=t[i]&&(a[i]=Oe((e[i]-t[i]+2*r)/s[i]+1,o));return a}function Xr(e,t,n,s=1){const r=be(t,s);return Math.floor((e[0]*(n-1)-n+r)/2)}function Pe(e){return typeof e=="number"?[e,e,e]:e.length===2?[e[0],e[1],1]:e}function Vn(e){return typeof e=="number"?[e,e,e]:e}function be(e,t){return t<=1?e:e+(e-1)*(t-1)}function Jl(e,t,n,s,r,o,a,i,c){let u,h,l;if(typeof e=="number"){u={top:e,bottom:e,left:e,right:e,type:e===0?"VALID":"NUMBER"};const g=Yl([t,n],o,s,e,i);h=g[0],l=g[1]}else if(e==="same"){h=Math.ceil(t/s),l=Math.ceil(n/r);const d=Math.max(0,(h-1)*s+o-t),g=Math.max(0,(l-1)*r+a-n),w=Math.floor(d/2),y=d-w,$=Math.floor(g/2),k=g-$;u={top:w,bottom:y,left:$,right:k,type:"SAME"}}else if(e==="valid")u={top:0,bottom:0,left:0,right:0,type:"VALID"},h=Math.ceil((t-o+1)/s),l=Math.ceil((n-a+1)/r);else if(typeof e=="object"){const d=c==="channelsLast"?e[1][0]:e[2][0],g=c==="channelsLast"?e[1][1]:e[2][1],w=c==="channelsLast"?e[2][0]:e[3][0],y=c==="channelsLast"?e[2][1]:e[3][1];u={top:d,bottom:g,left:w,right:y,type:d===0&&g===0&&w===0&&y===0?"VALID":"EXPLICIT"},h=Oe((t-o+d+g)/s+1,i),l=Oe((n-a+w+y)/r+1,i)}else throw Error(`Unknown padding parameter: ${e}`);return{padInfo:u,outHeight:h,outWidth:l}}function Ql(e,t,n,s,r,o,a,i,c,u,h){let l,d,g,w;if(e==="valid"&&(e=0),typeof e=="number"){l={top:e,bottom:e,left:e,right:e,front:e,back:e,type:e===0?"VALID":"NUMBER"};const $=Zl([t,n,s,1],[i,c,u],1,[r,o,a],e,h);d=$[0],g=$[1],w=$[2]}else if(e==="same"){d=Math.ceil(t/r),g=Math.ceil(n/o),w=Math.ceil(s/a);const y=(d-1)*r+i-t,$=(g-1)*o+c-n,k=(w-1)*a+u-s,I=Math.floor(y/2),D=y-I,E=Math.floor($/2),T=$-E,_=Math.floor(k/2),A=k-_;l={top:E,bottom:T,left:_,right:A,front:I,back:D,type:"SAME"}}else throw Error(`Unknown padding parameter: ${e}`);return{padInfo:l,outDepth:d,outHeight:g,outWidth:w}}function Oe(e,t){if(!t)return Math.trunc(e);switch(t){case"round":return Math.round(e);case"ceil":return Math.ceil(e);case"floor":return Math.floor(e);default:throw new Error(`Unknown roundingMode ${t}`)}}function un(e){const[t,n,s]=Pe(e);return t===1&&n===1&&s===1}function zt(e,t){return un(e)||un(t)}function Se(e){return Pe(e).every(t=>t>0)}function th(e){if(e==="NHWC")return"channelsLast";if(e==="NCHW")return"channelsFirst";throw new Error(`Unknown dataFormat ${e}`)}function Tt(e,t,n){if(n!=null){if(typeof t=="string")throw Error(`Error in ${e}: pad must be an integer when using dimRoundingMode ${n} but got pad ${t}.`);if(typeof t=="number")p($e(t),()=>`Error in ${e}: pad must be an integer when using dimRoundingMode ${n} but got pad ${t}.`);else if(typeof t=="object")t.forEach(s=>{s.forEach(r=>{p($e(r),()=>`Error in ${e}: pad must be an integer when using dimRoundingMode ${n} but got pad ${r}.`)})});else throw Error(`Error in ${e}: Unknown padding parameter: ${t}`)}}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function eh(e,t){const s={x:f(e,"x","reshape","string_or_numeric")},r={shape:t};return b.runKernel(pc,s,r)}const x=m({reshape_:eh});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function nh(e,t,n,s,r){const o=f(e,"x","avgPool","float32"),a=1;p(zt(n,a),()=>`Error in avgPool: Either strides or dilations must be 1. Got strides ${n} and dilations '${a}'`);let i=o,c=!1;o.rank===3&&(c=!0,i=x(o,[1,o.shape[0],o.shape[1],o.shape[2]])),p(i.rank===4,()=>`Error in avgPool: x must be rank 4 but got rank ${i.rank}.`),Tt("avgPool",s,r);const u={x:i},h={filterSize:t,strides:n,pad:s,dimRoundingMode:r};let l=b.runKernel(Da,u,h);return l=st(l,o.dtype),c?x(l,[l.shape[1],l.shape[2],l.shape[3]]):l}const Yr=m({avgPool_:nh});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function sh(e,t,n,s,r,o="NDHWC"){const a=f(e,"x","avgPool3d","float32");let i=a,c=!1;a.rank===4&&(c=!0,i=x(a,[1,a.shape[0],a.shape[1],a.shape[2],a.shape[3]])),p(i.rank===5,()=>`Error in avgPool3d: x must be rank 5 but got rank ${i.rank}.`),p(o==="NDHWC",()=>`Error in avgPool3d: Only NDHWC is currently supported, but got dataFormat of ${o}`),p(typeof n=="number"&&n>0||Array.isArray(n)&&n[0]>0&&n[1]>0&&n[2]>0,()=>`Error in avgPool3d: Stride must be > 0, but got '${n}'`),Tt("avgPool3d",s,r);const u={x:i},h={filterSize:t,strides:n,pad:s,dimRoundingMode:r,dataFormat:o};let l=b.runKernel(Na,u,h);return l=st(l,i.dtype),c?x(l,[l.shape[1],l.shape[2],l.shape[3],l.shape[4]]):l}const rh=m({avgPool3d_:sh});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function oh(e,t=0){p(e.length>=1,()=>"Pass at least one tensor to concat");const n=Re(e,"tensors","concat","string_or_numeric");if(n[0].dtype==="complex64"&&n.forEach(o=>{if(o.dtype!=="complex64")throw new Error(`Cannot concatenate complex64 tensors with a tensor
          with dtype ${o.dtype}. `)}),n.length===1)return Jt(n[0]);const s=n,r={axis:t};return b.runKernel(Ua,s,r)}const ft=m({concat_:oh});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function ah(e,t,n=!1,s=!1){let r=f(e,"a","matMul"),o=f(t,"b","matMul");[r,o]=Y(r,o);const a={a:r,b:o},i={transposeA:n,transposeB:s};return b.runKernel(Aa,a,i)}const O=m({matMul_:ah});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function ih(e){const n={x:f(e,"x","sigmoid","float32")};return b.runKernel(Nc,n)}const we=m({sigmoid_:ih});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function ch(e,t,n){const s=f(e,"x","slice","string_or_numeric");if(s.rank===0)throw new Error("Slicing scalar is not possible");const r={x:s},o={begin:t,size:n};return b.runKernel(Tc,r,o)}const j=m({slice_:ch});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function uh(e){const n={x:f(e,"x","tanh","float32")};return b.runKernel(Yc,n)}const jn=m({tanh_:uh});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function lh(e,t,n,s,r,o){const a=f(e,"forgetBias","basicLSTMCell"),i=f(t,"lstmKernel","basicLSTMCell"),c=f(n,"lstmBias","basicLSTMCell"),u=f(s,"data","basicLSTMCell"),h=f(r,"c","basicLSTMCell"),l=f(o,"h","basicLSTMCell"),d=ft([u,l],1),g=O(d,i),w=M(g,c),y=w.shape[0],$=w.shape[1]/4,k=[y,$],I=j(w,[0,0],k),D=j(w,[0,$],k),E=j(w,[0,$*2],k),T=j(w,[0,$*3],k),_=M(S(we(I),jn(D)),S(h,we(M(a,E)))),A=S(jn(_),we(T));return[_,A]}const hh=m({basicLSTMCell_:lh});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function fh(e,t,n){const s=f(e,"x","batchToSpaceND"),r=t.reduce((i,c)=>i*c);p(s.rank>=1+t.length,()=>`input rank is ${s.rank} but should be > than blockShape.length ${t.length}`),p(n.length===t.length,()=>`crops.length is ${n.length} but should be equal to blockShape.length  ${t.length}`),p(s.shape[0]%r===0,()=>`input tensor batch is ${s.shape[0]} but is not divisible by the product of the elements of blockShape ${t.join(" * ")} === ${r}`);const o={x:s},a={blockShape:t,crops:n};return b.runKernel(Ma,o,a)}const Zr=m({batchToSpaceND_:fh});function dh(e){let t;return e.rank===0||e.rank===1?t=x(e,[1,1,1,e.size]):e.rank===2?t=x(e,[1,1,e.shape[0],e.shape[1]]):e.rank===3?t=x(e,[1,e.shape[0],e.shape[1],e.shape[2]]):t=e,t}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function ph(e,t,n,s,r,o){o==null&&(o=.001);const a=f(e,"x","batchNorm"),i=f(t,"mean","batchNorm"),c=f(n,"variance","batchNorm");let u;r!=null&&(u=f(r,"scale","batchNorm"));let h;s!=null&&(h=f(s,"offset","batchNorm")),p(i.rank===c.rank,()=>"Batch normalization gradient requires mean and variance to have equal ranks."),p(h==null||i.rank===h.rank,()=>"Batch normalization gradient requires mean and offset to have equal ranks."),p(u==null||i.rank===u.rank,()=>"Batch normalization gradient requires mean and scale to have equal ranks.");const d={x:dh(a),scale:u,offset:h,mean:i,variance:c},g={varianceEpsilon:o},w=b.runKernel(bi,d,g);return x(w,a.shape)}const yn=m({batchNorm_:ph});function gh(e,t,n,s,r,o){const a=f(e,"x","batchNorm"),i=f(t,"mean","batchNorm"),c=f(n,"variance","batchNorm");let u;r!=null&&(u=f(r,"scale","batchNorm"));let h;return s!=null&&(h=f(s,"offset","batchNorm")),p(a.rank===2,()=>`Error in batchNorm2D: x must be rank 2 but got rank ${a.rank}.`),p(i.rank===2||i.rank===1,()=>`Error in batchNorm2D: mean must be rank 2 or rank 1 but got rank ${i.rank}.`),p(c.rank===2||c.rank===1,()=>`Error in batchNorm2D: variance must be rank 2 or rank 1 but got rank ${c.rank}.`),u!=null&&p(u.rank===2||u.rank===1,()=>`Error in batchNorm2D: scale must be rank 2 or rank 1 but got rank ${u.rank}.`),h!=null&&p(h.rank===2||h.rank===1,()=>`Error in batchNorm2D: offset must be rank 2 or rank 1 but got rank ${h.rank}.`),yn(a,i,c,h,u,o)}const mh=m({batchNorm2d_:gh});function bh(e,t,n,s,r,o){const a=f(e,"x","batchNorm"),i=f(t,"mean","batchNorm"),c=f(n,"variance","batchNorm");let u;r!=null&&(u=f(r,"scale","batchNorm"));let h;return s!=null&&(h=f(s,"offset","batchNorm")),p(a.rank===3,()=>`Error in batchNorm3D: x must be rank 3 but got rank ${a.rank}.`),p(i.rank===3||i.rank===1,()=>`Error in batchNorm3D: mean must be rank 3 or rank 1 but got rank ${i.rank}.`),p(c.rank===3||c.rank===1,()=>`Error in batchNorm3D: variance must be rank 3 or rank 1 but got rank ${c.rank}.`),u!=null&&p(u.rank===3||u.rank===1,()=>`Error in batchNorm3D: scale must be rank 3 or rank 1 but got rank ${u.rank}.`),h!=null&&p(h.rank===3||h.rank===1,()=>`Error in batchNorm3D: offset must be rank 3 or rank 1 but got rank ${h.rank}.`),yn(a,i,c,h,u,o)}const wh=m({batchNorm3d_:bh});function yh(e,t,n,s,r,o){const a=f(e,"x","batchNorm"),i=f(t,"mean","batchNorm"),c=f(n,"variance","batchNorm");let u;r!=null&&(u=f(r,"scale","batchNorm"));let h;return s!=null&&(h=f(s,"offset","batchNorm")),p(a.rank===4,()=>`Error in batchNorm4D: x must be rank 4 but got rank ${a.rank}.`),p(i.rank===4||i.rank===1,()=>`Error in batchNorm4D: mean must be rank 4 or rank 1 but got rank ${i.rank}.`),p(c.rank===4||c.rank===1,()=>`Error in batchNorm4D: variance must be rank 4 or rank 1 but got rank ${c.rank}.`),u!=null&&p(u.rank===4||u.rank===1,()=>`Error in batchNorm4D: scale must be rank 4 or rank 1 but got rank ${u.rank}.`),h!=null&&p(h.rank===4||h.rank===1,()=>`Error in batchNorm4D: offset must be rank 4 or rank 1 but got rank ${h.rank}.`),yn(a,i,c,h,u,o)}const $h=m({batchNorm4d_:yh});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function xh(e,t,n){const s=f(e,"x","bincount"),r=f(t,"weights","bincount");p(s.dtype==="int32",()=>`Error in bincount: input dtype must be int32, but got ${s.dtype}`),p(n>=0,()=>`size must be non-negative, but got ${n}.`),p(r.size===s.size||r.size===0,()=>`Error in bincount: weights must have the same size as input or0-length, but got input shape: ${s.shape}, weights shape: ${r.shape}.`);const o={x:s,weights:r},a={size:n};return b.runKernel(Fa,o,a)}const Jr=m({bincount_:xh});/**
 * @license
 * Copyright 2023 Google LLC.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function kh(e,t){const n=f(e,"x","bitwiseAnd"),s=f(t,"y","bitwiseAnd");if(!Wt(n.shape,s.shape))throw new Error(`BitwiseAnd: Tensors must have the same shape. x: ${n.shape}, y: ${s.shape}`);if(n.dtype!=="int32"||s.dtype!=="int32")throw new Error(`BitwiseAnd: Only supports 'int32' values in tensor, found type of x: ${n.dtype} and type of y: ${s.dtype}`);const r={a:n,b:s};return b.runKernel(Ba,r)}const vh=m({bitwiseAnd_:kh});/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Eh(e,t){const n=f(e,"s0","broadcastArgs","int32"),s=f(t,"s1","broadcastArgs","int32");if(n.rank!==1)throw new Error(`broadcastArgs(): first input must be a vector (rank=1). Has rank ${n.rank}`);if(s.rank!==1)throw new Error(`broadcastArgs(): second input must be a vector (rank=1). Has rank ${s.rank}`);const r={s0:n,s1:s};return b.runKernel(Ca,r)}const Sh=m({broadcastArgs_:Eh});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Th(e,t){let n=f(e,"broadcastTo","x");const s=n.shape;if(dt(t),t.length<n.rank)throw new Error(`broadcastTo(): shape.length=${t.length} < input.rank=${n.rank}.`);if(t.length>n.rank){const u=n.shape.slice();for(;u.length<t.length;)u.unshift(1);n=x(n,u)}const r=n.shape,o=Array.from(t);for(let u=t.length-1;u>=0;u--)if(r[u]===t[u])o[u]=1;else if(n.shape[u]!==1)throw new Error(`broadcastTo(): [${s}] cannot be broadcast to [${t}].`);if(o.map((u,h)=>u>1?h:-1).filter(u=>u>=0).length===0)return Jt(n);const i={x:n},c={reps:o};return b.runKernel(mr,i,c)}const nn=m({broadcastTo_:Th});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function _h(e){const n={x:f(e,"x","ceil","float32")};return b.runKernel(Ra,n)}const Ih=m({ceil_:_h});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Ye(e,t,n){dt(e),n=n||He(t);const s={shape:e,value:t,dtype:n};return b.runKernel(di,{},s)}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Dh(e,t,n){const s=f(e,"x","clipByValue");if(p(t<=n,()=>`Error in clip: min (${t}) must be less than or equal to max (${n}).`),t===n)return Ye(s.shape,t,s.dtype);const r={x:s},o={clipValueMin:t,clipValueMax:n};return b.runKernel(Pa,r,o)}const Nh=m({clipByValue_:Dh});function Ah(e){return ft(e,0)}const Mh=m({concat1d_:Ah});function Fh(e,t){return ft(e,t)}const Bh=m({concat2d_:Fh});function Ch(e,t){return ft(e,t)}const Rh=m({concat3d_:Ch});function Ph(e,t){return ft(e,t)}const Oh=m({concat4d_:Ph});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Lh(e,t,n,s,r="NHWC",o=[1,1],a){const i=f(e,"x","conv2d","float32"),c=f(t,"filter","conv2d","float32");let u=i,h=!1;i.rank===3&&(h=!0,u=x(i,[1,i.shape[0],i.shape[1],i.shape[2]])),p(u.rank===4,()=>`Error in conv2d: input must be rank 4, but got rank ${u.rank}.`),p(c.rank===4,()=>`Error in conv2d: filter must be rank 4, but got rank ${c.rank}.`),Tt("conv2d",s,a);const l=r==="NHWC"?u.shape[3]:u.shape[1];p(l===c.shape[2],()=>`Error in conv2d: depth of input (${l}) must match input depth for filter ${c.shape[2]}.`),p(zt(n,o),()=>`Error in conv2D: Either strides or dilations must be 1. Got strides ${n} and dilations '${o}'`),p(Se(o),()=>"Error in conv2D: Dilated rates should be larger than 0."),p(Se(n),()=>"Error in conv2D: Strides should be larger than 0.");const d={x:u,filter:c},g={strides:n,pad:s,dataFormat:r,dilations:o,dimRoundingMode:a},w=b.runKernel(Wa,d,g);return h?x(w,[w.shape[1],w.shape[2],w.shape[3]]):w}const $n=m({conv2d_:Lh});function Uh(e,t,n,s,r="NWC",o=1,a){const i=f(e,"x","conv1d"),c=f(t,"filter","conv1d");let u=i,h=!1;i.rank===2&&(h=!0,u=x(i,[1,i.shape[0],i.shape[1]])),p(u.rank===3,()=>`Error in conv1d: input must be rank 3, but got rank ${u.rank}.`),p(c.rank===3,()=>`Error in conv1d: filter must be rank 3, but got rank ${c.rank}.`),Tt("conv1d",s,a),p(u.shape[2]===c.shape[1],()=>`Error in conv1d: depth of input (${u.shape[2]}) must match input depth for filter ${c.shape[1]}.`),p(zt(n,o),()=>`Error in conv1D: Either stride or dilation must be 1. Got stride ${n} and dilation '${o}'`),p(Se(o),()=>"Error in conv1D: Dilated rates should be larger than 0."),p(Se(n),()=>"Error in conv1D: Stride should be larger than 0."),p(r==="NWC",()=>`Error in conv1d: got dataFormat of ${r} but only NWC is currently supported.`);const l=x(c,[1,c.shape[0],c.shape[1],c.shape[2]]),d=x(u,[u.shape[0],1,u.shape[1],u.shape[2]]),$=$n(d,l,[1,n],s,"NHWC",[1,o],a);return h?x($,[$.shape[2],$.shape[3]]):x($,[$.shape[0],$.shape[2],$.shape[3]])}const Wh=m({conv1d_:Uh});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function qh(e,t,n,s,r,o="NHWC",a){p(e.length===t.rank,()=>`Length of inShape (${e.length}) and rank of dy (${t.rank}) must match`);let i=e,c=t,u=!1;t.rank===3&&(u=!0,c=x(t,[1,t.shape[0],t.shape[1],t.shape[2]]),i=[1,e[0],e[1],e[2]]),p(i.length===4,()=>`Error in conv2dDerInput: inShape must be length 4, but got length ${i.length}.`),p(c.rank===4,()=>`Error in conv2dDerInput: dy must be rank 4, but got rank ${c.rank}`),p(n.rank===4,()=>`Error in conv2dDerInput: filter must be rank 4, but got rank ${n.rank}`);const h=o==="NHWC"?i[3]:i[1],l=o==="NHWC"?c.shape[3]:c.shape[1];p(h===n.shape[2],()=>`Error in conv2dDerInput: depth of input (${h}) must match input depth for filter ${n.shape[2]}.`),p(l===n.shape[3],()=>`Error in conv2dDerInput: depth of output (${l}) must match output depth for filter ${n.shape[3]}.`),Tt("conv2dDerInput",r,a);const d={dy:c,filter:n},g={strides:s,pad:r,dataFormat:o,dimRoundingMode:a,inputShape:i},w=b.runKernel(Ka,d,g);return u?x(w,[w.shape[1],w.shape[2],w.shape[3]]):w}const Qr=m({conv2DBackpropInput_:qh});function Kh(e,t,n,s,r,o){const a=f(e,"x","conv2dTranspose"),i=f(t,"filter","conv2dTranspose");return Qr(n,a,i,s,r,"NHWC",o)}const zh=m({conv2dTranspose_:Kh});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Gh(e,t,n,s,r="NDHWC",o=[1,1,1]){const a=f(e,"x","conv3d"),i=f(t,"filter","conv3d");let c=a,u=!1;a.rank===4&&(u=!0,c=x(a,[1,a.shape[0],a.shape[1],a.shape[2],a.shape[3]])),p(c.rank===5,()=>`Error in conv3d: input must be rank 5, but got rank ${c.rank}.`),p(i.rank===5,()=>`Error in conv3d: filter must be rank 5, but got rank ${i.rank}.`),p(c.shape[4]===i.shape[3],()=>`Error in conv3d: depth of input (${c.shape[4]}) must match input depth for filter ${i.shape[3]}.`),p(zt(n,o),()=>`Error in conv3D: Either strides or dilations must be 1. Got strides ${n} and dilations '${o}'`),p(r==="NDHWC",()=>`Error in conv3d: got dataFormat of ${r} but only NDHWC is currently supported.`),p(Se(o),()=>"Error in conv3D: Dilated rates should be larger than 0."),p(Se(n),()=>"Error in conv3D: Strides should be larger than 0.");const h={x:c,filter:i},l={strides:n,pad:s,dataFormat:r,dilations:o},d=b.runKernel(za,h,l);return u?x(d,[d.shape[1],d.shape[2],d.shape[3],d.shape[4]]):d}const Hh=m({conv3d_:Gh});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Vh(e,t,n,s,r){p(e.length===t.rank,()=>`Length of inShape (${e.length}) and rank of dy (${t.rank}) must match`);let o=e,a=t,i=!1;t.rank===4&&(i=!0,a=x(t,[1,t.shape[0],t.shape[1],t.shape[2],t.shape[3]]),o=[1,e[0],e[1],e[2],e[3]]);const c=o[4],u=a.shape[4];p(o.length===5,()=>`Error in conv3dDerInput: inShape must be length 5, but got length ${o.length}.`),p(a.rank===5,()=>`Error in conv3dDerInput: dy must be rank 5, but got rank ${a.rank}`),p(n.rank===5,()=>`Error in conv3dDerInput: filter must be rank 5, but got rank ${n.rank}`),p(c===n.shape[3],()=>`Error in conv3dDerInput: depth of input (${c}) must match input depth for filter ${n.shape[3]}.`),p(u===n.shape[4],()=>`Error in conv3dDerInput: depth of output (${u}) must match output depth for filter ${n.shape[4]}.`);const h={dy:a,filter:n},l={pad:r,strides:s,inputShape:o},d=b.runKernel(Ga,h,l);return i?x(d,[d.shape[1],d.shape[2],d.shape[3],d.shape[4]]):d}const jh=m({conv3DBackpropInput_:Vh});function Xh(e,t,n,s,r){const o=f(e,"x","conv3dTranspose"),a=f(t,"filter","conv3dTranspose");return jh(n,o,a,s,r)}const Yh=m({conv3dTranspose_:Xh});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Zh(e){const n={x:f(e,"x","cos","float32")};return b.runKernel(Ha,n)}const Jh=m({cos_:Zh});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Qh(e){const n={x:f(e,"x","cosh","float32")};return b.runKernel(Va,n)}const tf=m({cosh_:Qh});/**
 * @license
 * Copyright 2022 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the 'License');
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an 'AS IS' BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function ef(e,t=0,n=!1,s=!1){const o={x:f(e,"x","cumprod")},a={axis:t,exclusive:n,reverse:s};return b.runKernel(ja,o,a)}const nf=m({cumprod_:ef});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function sf(e,t=0,n=!1,s=!1){const o={x:f(e,"x","cumsum")},a={axis:t,exclusive:n,reverse:s};return b.runKernel(Xa,o,a)}const rf=m({cumsum_:sf});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function of(e,t,n,s=!1){const r=f(e,"x","denseBincount"),o=f(t,"weights","denseBincount");p(r.dtype==="int32",()=>`Error in denseBincount: input dtype must be int32, but got ${r.dtype}`),p(r.rank<=2,()=>`Error in denseBincount: input must be at most rank 2, but got rank ${r.rank}.`),p(n>=0,()=>`size must be non-negative, but got ${n}.`),p(o.size===r.size||o.size===0,()=>`Error in denseBincount: weights must have the same shape as x or 0-length, but got x shape: ${r.shape}, weights shape: ${o.shape}.`);const a={x:r,weights:o},i={size:n,binaryOutput:s};return b.runKernel(Za,a,i)}const af=m({denseBincount_:of});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function cf(e,t,n="NHWC"){const s=f(e,"x","depthToSpace","float32"),r=n==="NHWC"?s.shape[1]:s.shape[2],o=n==="NHWC"?s.shape[2]:s.shape[3],a=n==="NHWC"?s.shape[3]:s.shape[1];p(t>1,()=>`blockSize should be > 1 for depthToSpace, but was: ${t}`),p(r*t>=0,()=>`Negative dimension size caused by overflow when multiplying
    ${r} and ${t}  for depthToSpace with input shape
    ${s.shape}`),p(o*t>=0,()=>`Negative dimension size caused by overflow when multiplying
    ${o} and ${t} for depthToSpace with input shape
        ${s.shape}`),p(a%(t*t)===0,()=>`Dimension size must be evenly divisible by ${t*t} but is ${a} for depthToSpace with input shape ${s.shape}`);const i={x:s},c={blockSize:t,dataFormat:n};return b.runKernel(Ja,i,c)}const uf=m({depthToSpace_:cf});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function lf(e,t,n,s,r="NHWC",o=[1,1],a){const i=f(e,"x","depthwiseConv2d","float32"),c=f(t,"filter","depthwiseConv2d","float32");let u=i,h=!1;i.rank===3&&(h=!0,u=x(i,[1,i.shape[0],i.shape[1],i.shape[2]])),p(u.rank===4,()=>`Error in depthwiseConv2d: input must be rank 4, but got rank ${u.rank}.`),p(c.rank===4,()=>`Error in depthwiseConv2d: filter must be rank 4, but got rank ${c.rank}.`);const l=r==="NHWC"?u.shape[3]:u.shape[1];p(l===c.shape[2],()=>`Error in depthwiseConv2d: number of input channels (${l}) must match the inChannels dimension in filter ${c.shape[2]}.`),Tt("depthwiseConv2d",s,a);const d={x:u,filter:c},g={strides:n,pad:s,dataFormat:r,dilations:o,dimRoundingMode:a},w=b.runKernel(Qa,d,g);return h?x(w,[w.shape[1],w.shape[2],w.shape[3]]):w}const cs=m({depthwiseConv2d_:lf});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function hf(e){const n={x:f(e,"x","diag")};return b.runKernel(ni,n)}const ff=m({diag_:hf});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function df(e,t,n,s,r=[1,1],o="NHWC"){const a=f(e,"x","dilation2d"),i=f(t,"filter","dilation2d");p(a.rank===3||a.rank===4,()=>`Error in dilation2d: input must be rank 3 or 4, but got rank ${a.rank}.`),p(i.rank===3,()=>`Error in dilation2d: filter must be rank 3, but got rank ${i.rank}.`),p(o==="NHWC",()=>`Error in dilation2d: Only NHWC is currently supported, but got dataFormat of ${o}`);let c=a,u=!1;a.rank===3&&(c=x(a,[1,a.shape[0],a.shape[1],a.shape[2]]),u=!0),p(c.shape[3]===i.shape[2],()=>`Error in dilation2d:  input and filter must have the same depth: ${c.shape[3]} vs ${i.shape[2]}`);const h={x:c,filter:i},l={strides:n,pad:s,dilations:r},d=b.runKernel(si,h,l);return u?x(d,[d.shape[1],d.shape[2],d.shape[3]]):d}const pf=m({dilation2d_:df});/**
 * @license
 * Copyright 2017 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function gf(e,t){const n=e.length,s=[];for(let r=0;r<n;r++){const o=n-1-r,a=e[o]||1;(t[t.length-1-r]||1)>1&&a===1&&s.unshift(o)}return s}function to(e,t){const n=[];for(let s=0;s<t.length;s++){const r=e[e.length-s-1],o=t.length-s-1,a=t[o];(r==null||r===1&&a>1)&&n.unshift(o)}return n}function Q(e,t){const n=Math.max(e.length,t.length),s=new Array(n);for(let r=0;r<n;r++){let o=e[e.length-r-1];o==null&&(o=1);let a=t[t.length-r-1];if(a==null&&(a=1),o===1)s[n-r-1]=a;else if(a===1)s[n-r-1]=o;else if(o!==a){const i=`Operands could not be broadcast together with shapes ${e} and ${t}.`;throw Error(i)}else s[n-r-1]=o}return s}const X1=Object.freeze(Object.defineProperty({__proto__:null,assertAndGetBroadcastShape:Q,getBroadcastDims:gf,getReductionAxes:to},Symbol.toStringTag,{value:"Module"}));/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function mf(e,t){let n=f(e,"a","equal","string_or_numeric"),s=f(t,"b","equal","string_or_numeric");[n,s]=Y(n,s),Q(n.shape,s.shape);const r={a:n,b:s};return b.runKernel(ci,r)}const eo=m({equal_:mf});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function bf(e,t,n){const s=f(t,"a","where"),r=f(n,"b","where"),o=f(e,"condition","where","bool"),a=Q(Q(o.shape,s.shape),r.shape),i=nn(o,a),c=nn(s,a),u=nn(r,a),h={condition:i,t:c,e:u};return b.runKernel(Ec,h)}const Ot=m({where_:bf});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function wf(e){const n={x:f(e,"x","zerosLike")};return b.runKernel(nu,n)}const mt=m({zerosLike_:wf});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function yf(e,t){let n=f(e,"a","div"),s=f(t,"b","div");[n,s]=Y(n,s);const r=z(n,s),o=mt(r),a=eo(s,o);return Ot(a,o,r)}const $f=m({divNoNan_:yf});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function xf(e,t){const n=f(e,"t1","dot"),s=f(t,"t2","dot");p((n.rank===1||n.rank===2)&&(s.rank===1||s.rank===2),()=>`Error in dot: inputs must all be rank 1 or 2, but got ranks ${n.rank} and ${s.rank}.`);const r=n.rank===1?n.size:n.shape[1],o=s.rank===1?s.size:s.shape[0];if(p(r===o,()=>`Error in dot: inner dimensions of inputs must match, but got ${r} and ${o}.`),n.rank===1&&s.rank===1){const a=x(n,[1,-1]),i=x(s,[-1,1]),c=O(a,i);return x(c,[])}else if(n.rank===1&&s.rank===2){const a=x(n,[1,-1]),i=x(s,[s.shape[0],s.shape[1]]),c=O(a,i);return x(c,[c.size])}else if(n.rank===2&&s.rank===1){const a=x(s,[-1,1]),i=O(n,a);return x(i,[i.size])}else{const a=x(s,[s.shape[0],s.shape[1]]);return O(n,a)}}const kf=m({dot_:xf});/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function vf(e,...t){const n=t.map((r,o)=>f(r,`tensors${o}`,"einsum")),s={equation:e};return b.runKernel(oi,n,s)}const Ef=m({einsum_:vf});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Sf(e){const n={x:f(e,"x","elu","float32")};return b.runKernel(ai,n)}const no=m({elu_:Sf});/**
 * @license
 * Copyright 2023 Google LLC.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Tf(e,t){const n=f(e,"x","ensureShape","string_or_numeric");if(!nr(n.shape,t))throw new Error(`EnsureShape: Shape of tensor ${n.shape} is not compatible with expected shape ${t}`);return e}const _f=m({ensureShape_:Tf});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function If(e){let t=f(e,"x","erf");p(t.dtype==="int32"||t.dtype==="float32",()=>"Input dtype must be `int32` or `float32`."),t.dtype==="int32"&&(t=st(t,"float32"));const n={x:t};return b.runKernel(ii,n)}const Df=m({erf_:If});/**
 * @license
 * Copyright 2017 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function so(e,t){for(let n=0;n<e.length;++n)if(e[e.length-n-1]!==t-1-n)return!1;return!0}function Nf(e,t,n){const s=e.length+t.length,r=[];let o=0,a=0;for(let i=0;i<s;i++)n.indexOf(i)===-1?r.push(e[o++]):r.push(t[a++]);return r}function Y1(e,t){const n=[],s=e.length;for(let o=0;o<s;o++)t.indexOf(o)===-1&&n.push(e[o]);const r=t.map(o=>e[o]);return[n,r]}function xn(e,t){const n=t.map(s=>1);return Nf(e,n,t)}function Z1(e,t,n){p(so(t,n),()=>`${e} supports only inner-most axes for now. Got axes ${t} and rank-${n} input.`)}function J1(e,t){if(so(e,t))return null;const n=[];for(let s=0;s<t;++s)e.indexOf(s)===-1&&n.push(s);return e.forEach(s=>n.push(s)),n}function Q1(e){return e.map((t,n)=>[n,t]).sort((t,n)=>t[1]-n[1]).map(t=>t[0])}function t0(e,t){const n=[];for(let s=t-e;s<t;++s)n.push(s);return n}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Af(e,t=null,n=!1){const r={x:f(e,"x","max")},o={reductionIndices:t,keepDims:n};return b.runKernel(Pi,r,o)}const ye=m({max_:Af});/**
 * @license
 * Copyright 2020 Google Inc. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Mf(e,t=null,n=!1){const r={x:f(e,"x","min")},o={axis:t,keepDims:n};return b.runKernel(Ki,r,o)}const Xn=m({min_:Mf});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Ff(e,t){let n=f(e,"base","pow"),s=f(t,"exp","pow");[n,s]=Y(n,s);const r={a:n,b:s};return b.runKernel(rc,r)}const Le=m({pow_:Ff});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function U(e,t){if((bt(e)&&t!=="string"||Array.isArray(e))&&t!=="complex64")throw new Error("Error creating a new Scalar: value must be a primitive (number|boolean|string)");if(t==="string"&&bt(e)&&!(e instanceof Uint8Array))throw new Error("When making a scalar from encoded string, the value must be `Uint8Array`.");return Kt(e,[],[],t)}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Bf(e){const n={x:f(e,"x","sqrt","float32")};return b.runKernel(Mc,n)}const Mt=m({sqrt_:Bf});/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Cf(e){const t=f(e,"x","square"),n={};return b.runKernel("Square",{x:t},n)}const Et=m({square_:Cf});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Rf(e,t=null,n=!1){let s=f(e,"x","sum");s.dtype==="bool"&&(s=st(s,"int32"));const r={x:s},o={axis:t,keepDims:n};return b.runKernel(Fc,r,o)}const K=m({sum_:Rf});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Pf(e,t="euclidean",n=null,s=!1){e=f(e,"x","norm");const r=ro(e,t,n);let o=r.shape;if(s){const a=Ge(n,e.shape);o=xn(r.shape,a)}return x(r,o)}function ro(e,t,n=null){if(e.rank===0)return gt(e);if(e.rank!==1&&n===null)return ro(x(e,[-1]),t,n);if(e.rank===1||typeof n=="number"||Array.isArray(n)&&n.length===1){if(t===1)return K(gt(e),n);if(t===1/0)return ye(gt(e),n);if(t===-1/0)return Xn(gt(e),n);if(t==="euclidean"||t===2)return Mt(K(Le(gt(e),U(2,"int32")),n));throw new Error(`Error in norm: invalid ord value: ${t}`)}if(Array.isArray(n)&&n.length===2){if(t===1)return ye(K(gt(e),n[0]),n[1]-1);if(t===1/0)return ye(K(gt(e),n[1]),n[0]);if(t===-1/0)return Xn(K(gt(e),n[1]),n[0]);if(t==="fro"||t==="euclidean")return Mt(K(Et(e),n));throw new Error(`Error in norm: invalid ord value: ${t}`)}throw new Error(`Error in norm: invalid axis: ${n}`)}const kn=m({norm_:Pf});/**
 * @license
 * Copyright 2022 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Of(e,t=null,n=!1){return kn(e,"euclidean",t,n)}const Lf=m({euclideanNorm_:Of});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Uf(e){const n={x:f(e,"x","exp")};return b.runKernel(ui,n)}const ee=m({exp_:Uf});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Wf(e,t=0){const n=f(e,"x","expandDims","string_or_numeric");p(t<=n.rank,()=>"Axis must be <= rank of the tensor");const s={input:n},r={dim:t};return b.runKernel(li,s,r)}const Ht=m({expandDims_:Wf});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function qf(e){const n={x:f(e,"x","expm1")};return b.runKernel(hi,n)}const Kf=m({expm1_:qf});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function zf(e,t){const n=f(e,"x","tile","string_or_numeric");p(n.rank===t.length,()=>`Error in transpose: rank of input ${n.rank} must match length of reps ${t}.`);const s={x:n},r={reps:t};return b.runKernel(mr,s,r)}const Me=m({tile_:zf});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Gf(e,t,n,s="float32"){t==null&&(t=e);const r=At([e,t],s),o=e<=t?e:t;for(let i=0;i<o;++i)r.set(1,i,i);const a=x(r.toTensor(),[e,t]);if(n==null)return a;if(n.length===1)return Me(Ht(a,0),[n[0],1,1]);if(n.length===2)return Me(Ht(Ht(a,0),0),[n[0],n[1],1,1]);if(n.length===3)return Me(Ht(Ht(Ht(a,0),0),0),[n[0],n[1],n[2],1,1]);throw new Error(`eye() currently supports only 1D and 2D batchShapes, but received ${n.length}D.`)}const oo=m({eye_:Gf});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Hf(e){const n={x:f(e,"x","floor","float32")};return b.runKernel(gi,n)}const ao=m({floor_:Hf});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Vf(e,t,n=0,s=0){const r=f(e,"x","gather"),o=f(t,"indices","gather","int32"),a={x:r,indices:o},i={axis:n,batchDims:s};return b.runKernel(wi,a,i)}const io=m({gather_:Vf});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function jf(e,t){let n=f(e,"a","greater","string_or_numeric"),s=f(t,"b","greater","string_or_numeric");[n,s]=Y(n,s),Q(n.shape,s.shape);const r={a:n,b:s};return b.runKernel($i,r)}const vn=m({greater_:jf});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Xf(e,t){let n=f(e,"a","greaterEqual","string_or_numeric"),s=f(t,"b","greaterEqual","string_or_numeric");[n,s]=Y(n,s),Q(n.shape,s.shape);const r={a:n,b:s};return b.runKernel(xi,r)}const co=m({greaterEqual_:Xf});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Yf(e){const n={input:f(e,"input","imag")};return b.runKernel(vi,n)}const En=m({imag_:Yf});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Zf(e){const n={x:f(e,"x","isFinite")};return b.runKernel(Ei,n)}const Jf=m({isFinite_:Zf});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Qf(e){const n={x:f(e,"x","isInf")};return b.runKernel(Si,n)}const td=m({isInf_:Qf});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function ed(e){const n={x:f(e,"x","isNaN")};return b.runKernel(Ti,n)}const nd=m({isNaN_:ed});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function sd(e,t=.2){const s={x:f(e,"x","leakyRelu")},r={alpha:t};return b.runKernel(_i,s,r)}const uo=m({leakyRelu_:sd});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function rd(e,t){let n=f(e,"a","less","string_or_numeric"),s=f(t,"b","less","string_or_numeric");[n,s]=Y(n,s),Q(n.shape,s.shape);const r={a:n,b:s};return b.runKernel(Ii,r)}const Yn=m({less_:rd});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function od(e,t){let n=f(e,"a","lessEqual","string_or_numeric"),s=f(t,"b","lessEqual","string_or_numeric");[n,s]=Y(n,s),Q(n.shape,s.shape);const r={a:n,b:s};return b.runKernel(Di,r)}const us=m({lessEqual_:od});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function ad(e,t,n){if(n<=0)throw new Error("The number of values should be positive.");const s={start:e,stop:t,num:n};return b.runKernel(Ni,{},s)}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function id(e,t=5,n=1,s=1,r=.5){const o=f(e,"x","localResponseNormalization");p(o.rank===4||o.rank===3,()=>`Error in localResponseNormalization: x must be rank 3 or 4 but got
               rank ${o.rank}.`),p($e(t),()=>`Error in localResponseNormalization: depthRadius must be an integer but got depthRadius ${t}.`);let a=o,i=!1;o.rank===3&&(i=!0,a=x(o,[1,o.shape[0],o.shape[1],o.shape[2]]));const c={x:a},u={depthRadius:t,bias:n,alpha:s,beta:r},h=b.runKernel(Ri,c,u);return i?x(h,[h.shape[1],h.shape[2],h.shape[3]]):h}const cd=m({localResponseNormalization_:id});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function ud(e){const n={x:f(e,"x","log","float32")};return b.runKernel(Ai,n)}const Ue=m({log_:ud});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function ld(e){const n={x:f(e,"x","log1p")};return b.runKernel(Mi,n)}const lo=m({log1p_:ld});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function e0(e){return p(Lt(e),()=>"The f passed in grad(f) must be a function"),(t,n)=>{const s=f(t,"x","tf.grad","string_or_numeric"),r=n!=null?f(n,"dy","tf.grad"):null;return b.tidy(()=>{const{value:o,grads:a}=b.gradients(()=>e(s),[s],r);return r!=null&&at(o.shape,r.shape,"The shape of dy passed in grad(f)(x, dy) must match the shape returned by f(x)"),Sn(a),a[0]})}}function n0(e){return p(Lt(e),()=>"The f passed in grads(f) must be a function"),(t,n)=>{p(Array.isArray(t),()=>"The args passed in grads(f)(args) must be an array of `Tensor`s or `TensorLike`s");const s=Re(t,"args","tf.grads","string_or_numeric"),r=n!=null?f(n,"dy","tf.grads"):null;return b.tidy(()=>{const{value:o,grads:a}=b.gradients(()=>e(...s),s,r);return r!=null&&at(o.shape,r.shape,"The shape of dy passed in grads(f)([x1,...], dy) must match the shape returned by f([x1,...])"),Sn(a),a})}}function s0(e){return p(Lt(e),()=>"The f passed in valueAndGrad(f) must be a function"),(t,n)=>{p(t instanceof nt,()=>"The x passed in valueAndGrad(f)(x) must be a tensor"),p(n==null||n instanceof nt,()=>"The dy passed in valueAndGrad(f)(x, dy) must be a tensor");const{grads:s,value:r}=b.gradients(()=>e(t),[t],n);return Sn(s),{grad:s[0],value:r}}}function r0(e){return p(Lt(e),()=>"The f passed in valueAndGrads(f) must be a function"),(t,n)=>{p(Array.isArray(t)&&t.every(r=>r instanceof nt),()=>"The args passed in valueAndGrads(f)(args) must be array of tensors"),p(n==null||n instanceof nt,()=>"The dy passed in valueAndGrads(f)(args, dy) must be a tensor");const s=b.gradients(()=>e(...t),t,n);return n!=null&&at(s.value.shape,n.shape,"The shape of dy passed in valueAndGrads(f)([x1,...], dy) must match the shape returned by f([x1,...])"),Sn(s.grads),s}}function hd(e,t){p(Lt(e),()=>"The f passed in variableGrads(f) must be a function"),p(t==null||Array.isArray(t)&&t.every(u=>u instanceof an),()=>"The varList passed in variableGrads(f, varList) must be an array of variables");const n=t!=null;if(!n){t=[];for(const u in b.registeredVariables)t.push(b.registeredVariables[u])}const s=n?t.filter(u=>!u.trainable):null,r=t.length;t=t.filter(u=>u.trainable),p(t.length>0,()=>`variableGrads() expects at least one of the input variables to be trainable, but none of the ${r} variables is trainable.`);const o=!0,{value:a,grads:i}=b.gradients(e,t,null,o);p(i.some(u=>u!=null),()=>"Cannot find a connection between any variable and the result of the loss function y=f(x). Please make sure the operations that use variables are inside the function f passed to minimize()."),p(a.rank===0,()=>`The f passed in variableGrads(f) must return a scalar, but it returned a rank-${a.rank} tensor`);const c={};return t.forEach((u,h)=>{i[h]!=null&&(c[u.name]=i[h])}),s!=null&&s.forEach(u=>c[u.name]=null),{value:a,grads:c}}function Ft(e){return b.customGrad(e)}function Sn(e){if(e.filter(n=>n==null).length>0)throw new Error(`Cannot compute gradient of y=f(x) with respect to x. Make sure that
    the f you passed encloses all operations that lead from x to y.`)}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function fd(e){const n={x:f(e,"x","neg")};return b.runKernel(Xi,n)}const _t=m({neg_:fd});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function dd(e){const n={x:f(e,"x","softplus")};return b.runKernel(Ac,n)}const ho=m({softplus_:dd});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function pd(e){const t=f(e,"x","logSigmoid");return Ft(s=>({value:_t(ho(_t(s))),gradFunc:a=>S(a,we(_t(s)))}))(t)}const gd=m({logSigmoid_:pd});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function md(e,t){let n=f(e,"a","sub"),s=f(t,"b","sub");[n,s]=Y(n,s);const r={a:n,b:s};return b.runKernel(jc,r)}const R=m({sub_:md});/**
 * @license
 * Copyright 2020 Google Inc. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function bd(e,t=-1){const n=f(e,"logits","logSoftmax");if(t===-1&&(t=n.rank-1),t!==n.rank-1)throw Error(`Log Softmax along a non-last dimension is not yet supported. Logits was rank ${n.rank} and axis was ${t}`);return Ft((r,o)=>{const i=ye(r,t,!0),c=R(r,i),u=R(st(c,"float32"),Ue(K(ee(c),t,!0)));return o([u]),{value:u,gradFunc:(l,d)=>{const[g]=d,w=!0,y=ee(g);return R(l,S(K(l,t,w),y))}}})(n)}const wd=m({logSoftmax_:bd});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function yd(e,t=null,n=!1){const s=f(e,"x","logSumExp"),r=Ge(t,s.shape),o=ye(s,r,!0),a=R(s,o),i=ee(a),c=K(i,r),u=Ue(c),h=M(x(o,u.shape),u);if(n){const l=xn(h.shape,r);return x(h,l)}return h}const fo=m({logSumExp_:yd});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function $d(e,t){const n=f(e,"a","logicalAnd","bool"),s=f(t,"b","logicalAnd","bool");Q(n.shape,s.shape);const r={a:n,b:s};return b.runKernel(Fi,r)}const ln=m({logicalAnd_:$d});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function xd(e){const n={x:f(e,"x","logicalNot","bool")};return b.runKernel(Bi,n)}const po=m({logicalNot_:xd});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function kd(e,t){const n=f(e,"a","logicalOr","bool"),s=f(t,"b","logicalOr","bool");Q(n.shape,s.shape);const r={a:n,b:s};return b.runKernel(Ci,r)}const go=m({logicalOr_:kd});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function vd(e,t){const n=f(e,"a","logicalXor","bool"),s=f(t,"b","logicalXor","bool");return Q(n.shape,s.shape),ln(go(e,t),po(ln(e,t)))}const Ed=m({logicalXor_:vd});/**
 * @license
 * Copyright 2022 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Qe=2147483648;function Sd(e,t,n="left"){const s=f(e,"sortedSequence","searchSorted"),r=f(t,"values","searchSorted"),o=s.shape[s.shape.length-1],a=r.shape[r.shape.length-1],i=x(s,[-1,o]),c=x(r,[-1,a]);if(i.rank<2)throw new Error("Sorted input argument must be at least 2-dimensional");if(i.shape[0]!==c.shape[0])throw new Error("Leading dimension of 'sortedSequence' and 'values' must match.");if(Z(c.shape)>=Qe)throw new Error(`values tensor size must less than ${Qe}`);if(i.shape[1]>=Qe)throw new Error(`trailing dim_size must less than ${Qe} for int32 output type, was ${i.shape[1]}`);const u={sortedSequence:i,values:c},h={side:n};return b.runKernel(vc,u,h)}const ls=m({searchSorted_:Sd});/**
 * @license
 * Copyright 2022 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Td(e,t){return ls(e,t,"left")}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function _d(e,t,n,s,r){const o=f(e,"x","maxPool"),a=1;let i=o,c=!1;o.rank===3&&(c=!0,i=x(o,[1,o.shape[0],o.shape[1],o.shape[2]])),p(i.rank===4,()=>`Error in maxPool: input must be rank 4 but got rank ${i.rank}.`),p(zt(n,a),()=>`Error in maxPool: Either strides or dilations must be 1. Got strides ${n} and dilations '${a}'`),Tt("maxPool",s,r);const u={x:i},h={filterSize:t,strides:n,pad:s,dimRoundingMode:r},l=b.runKernel(Li,u,h);return c?x(l,[l.shape[1],l.shape[2],l.shape[3]]):l}const mo=m({maxPool_:_d});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Id(e,t=[1,1,1],n,s,r,o="NDHWC"){const a=f(e,"x","maxPool3d");let i=a,c=!1;a.rank===4&&(c=!0,i=x(a,[1,a.shape[0],a.shape[1],a.shape[2],a.shape[3]])),p(i.rank===5,()=>`Error in maxPool3d: x must be rank 5 but got rank ${i.rank}.`),p(o==="NDHWC",()=>`Error in maxPool3d: Only NDHWC is currently supported, but got dataFormat of ${o}`),Tt("maxPool3d",s,r);const u={x:i},h={filterSize:t,strides:n,pad:s,dimRoundingMode:r,dataFormat:o},l=b.runKernel(Ui,u,h);return c?x(l,[l.shape[1],l.shape[2],l.shape[3],l.shape[4]]):l}const Dd=m({maxPool3d_:Id});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Nd(e,t,n,s,r=!1){const a={x:f(e,"x","maxPoolWithArgmax")},i={filterSize:t,strides:n,pad:s,includeBatchInIndex:r},c=b.runKernel(Wi,a,i);return{result:c[0],indexes:c[1]}}const Ad=m({maxPoolWithArgmax_:Nd});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Md(e,t){let n=f(e,"a","maximum"),s=f(t,"b","maximum");[n,s]=Y(n,s),n.dtype==="bool"&&(n=st(n,"int32"),s=st(s,"int32")),Q(n.shape,s.shape);const r={a:n,b:s};return b.runKernel(Oi,r)}const bo=m({maximum_:Md});/**
 * @license
 * Copyright 2020 Google Inc. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Fd(e,t=null,n=!1){const r={x:f(e,"x","mean")},o={axis:t,keepDims:n};return b.runKernel(qi,r,o)}const hn=m({mean_:Fd});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Te(e,t="float32"){if(dt(e),t==="complex64"){const s=Te(e,"float32"),r=Te(e,"float32");return Ut(s,r)}const n=gn(Z(e),t);return b.makeTensor(n,e,t)}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Zt(e,t="float32"){if(dt(e),t==="complex64"){const s=Zt(e,"float32"),r=Te(e,"float32");return Ut(s,r)}const n=ts(Z(e),t);return b.makeTensor(n,e,t)}/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Bd(e,t,{indexing:n="xy"}={}){if(n!=="xy"&&n!=="ij")throw new TypeError(`${n} is not a valid third argument to meshgrid`);if(e===void 0)return[];let s=f(e,"x","meshgrid",e instanceof nt?e.dtype:"float32");if(t===void 0)return[s];let r=f(t,"y","meshgrid",t instanceof nt?t.dtype:"float32");const o=Z(s.shape),a=Z(r.shape);return n==="xy"?(s=x(s,[1,-1]),r=x(r,[-1,1]),[O(Zt([a,1],s.dtype),s),O(r,Zt([1,o],r.dtype))]):(s=x(s,[-1,1]),r=x(r,[1,-1]),[O(s,Zt([1,a],s.dtype)),O(Zt([o,1],r.dtype),r)])}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Cd(e,t){let n=f(e,"a","minimum"),s=f(t,"b","minimum");[n,s]=Y(n,s),n.dtype==="bool"&&(n=st(n,"int32"),s=st(s,"int32")),Q(n.shape,s.shape);const r={a:n,b:s};return b.runKernel(zi,r)}const fn=m({minimum_:Cd});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Rd(e,t,n){p(n==="reflect"||n==="symmetric",()=>`Invalid mode. Mode must be either reflect or symmetric. Got ${n}.`);const s=f(e,"x","mirrorPad");if(s.rank===0)throw new Error("mirrorPad(scalar) is not defined. Pass non-scalar to mirrorPad");p(t.length===s.rank,()=>`Padding doesn't match input. Must be ${s.rank}. Got ${t.length}.`);const r=n==="reflect"?1:0;for(let i=0;i<s.rank;i++)p(t[i].length===2,()=>"Invalid number of paddings. Must be length of 2 each."),p(t[i][0]>=0&&t[i][0]<=s.shape[i]-r&&t[i][1]>=0&&t[i][1]<=s.shape[i]-r,()=>`Padding in dimension ${i} cannot be greater than or equal to ${s.shape[i]-r} or less than 0 for input of shape ${s.shape}`);const o={paddings:t,mode:n},a={x:s};return b.runKernel(Gi,a,o)}const Pd=m({mirrorPad_:Rd});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Od(e,t){let n=f(e,"a","mod"),s=f(t,"b","mod");[n,s]=Y(n,s);const r={a:n,b:s};return b.runKernel(Hi,r)}const Ld=m({mod_:Od});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Ud(e,t=null,n=!1){e=f(e,"x","moments");const s=Ge(t,e.shape),r=hn(e,s,n);let o=r.shape;n||(o=xn(r.shape,s));const a=Et(R(st(e,"float32"),x(r,o))),i=hn(a,s,n);return{mean:r,variance:i}}const Wd=m({moments_:Ud});function qd(e,t,n,s){const r=f(t,"data","multiRNNCell"),o=Re(n,"c","multiRNNCell"),a=Re(s,"h","multiRNNCell");let i=r;const c=[];for(let l=0;l<e.length;l++){const d=e[l](i,o[l],a[l]);c.push(d[0]),c.push(d[1]),i=d[1]}const u=[],h=[];for(let l=0;l<c.length;l+=2)u.push(c[l]),h.push(c[l+1]);return[u,h]}const Kd=m({multiRNNCell_:qd});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function zd(e,t,n,s=!1){const r=f(e,"logits","multinomial"),o=r.size,a=r.rank;if(o<2)throw new Error(`Error in multinomial: you need at least 2 outcomes, but got ${o}.`);if(a>2)throw new Error(`Rank of probabilities must be 1 or 2, but is ${a}`);n=n||Math.random();const c={logits:a===1?x(r,[1,-1]):r},u={numSamples:t,seed:n,normalized:s},h=b.runKernel(Vi,c,u);return a===1?x(h,[h.size]):h}const Gd=m({multinomial_:zd});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Hd(e,t){let n=f(e,"a","notEqual","string_or_numeric"),s=f(t,"b","notEqual","string_or_numeric");[n,s]=Y(n,s),Q(n.shape,s.shape);const r={a:n,b:s};return b.runKernel(Yi,r)}const wo=m({notEqual_:Hd});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Vd(e,t,n=1,s=0,r="int32"){if(t<2)throw new Error(`Error in oneHot: depth must be >=2, but it is ${t}`);const a={indices:f(e,"indices","oneHot","int32")},i={dtype:r,depth:t,onValue:n,offValue:s};return b.runKernel(ec,a,i)}const jd=m({oneHot_:Vd});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Xd(e){const n={x:f(e,"x","onesLike")};return b.runKernel(tc,n)}const Yd=m({onesLike_:Xd});function Zd(e,t){const n=f(e,"v1","outerProduct"),s=f(t,"v2","outerProduct");p(n.rank===1&&s.rank===1,()=>`Error in outerProduct: inputs must be rank 1, but got ranks ${n.rank} and ${s.rank}.`);const r=x(n,[-1,1]),o=x(s,[1,-1]);return O(r,o)}const Jd=m({outerProduct_:Zd});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Qd(e,t,n=0){const s=f(e,"x","pad");if(s.rank===0)throw new Error("pad(scalar) is not defined. Pass non-scalar to pad");const r={paddings:t,constantValue:n},o={x:s};return b.runKernel(sc,o,r)}const Ze=m({pad_:Qd});function tp(e,t,n=0){return p(t.length===2,()=>"Invalid number of paddings. Must be length of 2."),Ze(e,[t],n)}const ep=m({pad1d_:tp});function np(e,t,n=0){return p(t.length===2&&t[0].length===2&&t[1].length===2,()=>"Invalid number of paddings. Must be length of 2 each."),Ze(e,t,n)}const sp=m({pad2d_:np});function rp(e,t,n=0){return p(t.length===3&&t[0].length===2&&t[1].length===2&&t[2].length===2,()=>"Invalid number of paddings. Must be length of 2 each."),Ze(e,t,n)}const op=m({pad3d_:rp});function ap(e,t,n=0){return p(t.length===4&&t[0].length===2&&t[1].length===2&&t[2].length===2&&t[3].length===2,()=>"Invalid number of paddings. Must be length of 2 each."),Ze(e,t,n)}const ip=m({pad4d_:ap});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function cp(e,t,n){const s=f(e,"x","spaceToBatchND");p(s.rank>=1+t.length,()=>`input rank ${s.rank} should be > than [blockShape] ${t.length}`),p(n.length===t.length,()=>`paddings.shape[0] ${n.length} must be equal to [blockShape] ${t.length}`),p(s.shape.reduce((a,i,c)=>c>0&&c<=t.length?a&&(i+n[c-1][0]+n[c-1][1])%t[c-1]===0:a,!0),()=>`input spatial dimensions ${s.shape.slice(1)} with paddings ${n.toString()} must be divisible by blockShapes ${t.toString()}`);const r={x:s},o={blockShape:t,paddings:n};return b.runKernel(Bc,r,o)}const yo=m({spaceToBatchND_:cp});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function up(e,t,n,s,r,o,a){r==null&&(r=[1,1]),o==null&&(o=1),s===0&&(s="valid");const i=f(e,"x","maxPool");let c=i,u=!1;i.rank===3&&(u=!0,c=x(i,[1,i.shape[0],i.shape[1],i.shape[2]])),p(zt(o,r),()=>`Error in pool: Either strides or dilations must be 1. Got strides ${o} and dilations '${r}'`);const h=jl(c.shape,t,o,r,s),l=[h.dilationHeight,h.dilationWidth];let d;s==="same"?d=hp([h.filterHeight,h.filterWidth],l):d=[[0,0],[0,0]];const g=l[0]===1&&l[1]===1,[w,y]=lp([h.inHeight,h.inWidth],l,d),$=g?s:"valid",k=g?c:yo(c,l,w),D=(n==="avg"?()=>Yr(k,t,o,$,a):()=>mo(k,t,o,$,a))(),E=g?D:Zr(D,l,y);return u?x(E,[E.shape[1],E.shape[2],E.shape[3]]):E}function lp(e,t,n){const s=n.map(h=>h[0]),r=n.map(h=>h[1]),o=e.concat(s,r),a=t.map((h,l)=>(h-o[l]%h)%h),i=r.map((h,l)=>h+a[l]),c=t.map((h,l)=>[s[l],i[l]]),u=t.map((h,l)=>[0,a[l]]);return[c,u]}function hp(e,t){const s=e.map((a,i)=>a+(a-1)*(t[i]-1)).map(a=>a-1),r=s.map(a=>Math.floor(a/2)),o=s.map((a,i)=>a-r[i]);return s.map((a,i)=>[r[i],o[i]])}const fp=m({pool_:up});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function dp(e,t){const n=f(e,"x","prelu"),s=f(t,"alpha","prelu"),r={x:n,alpha:s};return b.runKernel(oc,r)}const $o=m({prelu_:dp});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function pp(e,t=null,n=!1){let s=f(e,"x","prod");s.dtype==="bool"&&(s=st(s,"int32"));const r={x:s},o={axis:t,keepDims:n};return b.runKernel(ac,r,o)}const gp=m({prod_:pp});/**
 * @license
 * Copyright 2022 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function mp(e,t,n,s){const r=e.map((h,l)=>f(h,`tensors${l}`,"raggedGather","int32")),o=f(t,"paramsDenseValues","raggedGather"),a=f(n,"indices","raggedGather","int32"),i={paramsNestedSplits:r,paramsDenseValues:o,indices:a},c={outputRaggedRank:s},u=b.runKernel(ic,i,c);return{outputNestedSplits:u.slice(0,u.length-1),outputDenseValues:u[u.length-1]}}const bp=m({raggedGather_:mp});/**
 * @license
 * Copyright 2022 Google LLC.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function wp(e,t,n){const s=f(e,"starts","raggedRange"),r=f(t,"limits","raggedRange",s.dtype),o=f(n,"deltas","raggedRange",s.dtype),a={starts:s,limits:r,deltas:o},i=b.runKernel(cc,a);return{rtNestedSplits:i[0],rtDenseValues:i[1]}}const yp=m({raggedRange_:wp});/**
 * @license
 * Copyright 2022 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function $p(e,t,n,s,r){const o=f(e,"shape","raggedTensorToTensor","int32"),a=f(t,"values","raggedTensorToTensor"),i=f(n,"defaultValue","raggedTensorToTensor",a.dtype),c=s.map((l,d)=>f(l,`tensors${d}`,"raggedTensorToTensor","int32")),u={shape:o,values:a,defaultValue:i,rowPartitionTensors:c},h={rowPartitionTypes:r};return b.runKernel(uc,u,h)}const xp=m({raggedTensorToTensor_:$p});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function kp(e,t,n){dt(e);const s=Z(e);let r=null;if(n==null||n==="float32")r=new Float32Array(s);else if(n==="int32")r=new Int32Array(s);else if(n==="bool")r=new Uint8Array(s);else throw new Error(`Unknown data type ${n}`);for(let o=0;o<s;o++)r[o]=t();return b.makeTensor(r,e,n)}const vp=m({rand_:kp});var hs={exports:{}};hs.exports;(function(e){(function(t,n,s){function r(c){var u=this,h=i();u.next=function(){var l=2091639*u.s0+u.c*23283064365386963e-26;return u.s0=u.s1,u.s1=u.s2,u.s2=l-(u.c=l|0)},u.c=1,u.s0=h(" "),u.s1=h(" "),u.s2=h(" "),u.s0-=h(c),u.s0<0&&(u.s0+=1),u.s1-=h(c),u.s1<0&&(u.s1+=1),u.s2-=h(c),u.s2<0&&(u.s2+=1),h=null}function o(c,u){return u.c=c.c,u.s0=c.s0,u.s1=c.s1,u.s2=c.s2,u}function a(c,u){var h=new r(c),l=u&&u.state,d=h.next;return d.int32=function(){return h.next()*4294967296|0},d.double=function(){return d()+(d()*2097152|0)*11102230246251565e-32},d.quick=d,l&&(typeof l=="object"&&o(l,h),d.state=function(){return o(h,{})}),d}function i(){var c=4022871197,u=function(h){h=String(h);for(var l=0;l<h.length;l++){c+=h.charCodeAt(l);var d=.02519603282416938*c;c=d>>>0,d-=c,d*=c,c=d>>>0,d-=c,c+=d*4294967296}return(c>>>0)*23283064365386963e-26};return u}n&&n.exports?n.exports=a:s&&s.amd?s(function(){return a}):this.alea=a})(oe,e,!1)})(hs);var Ep=hs.exports,fs={exports:{}};fs.exports;(function(e){(function(t,n,s){function r(i){var c=this,u="";c.x=0,c.y=0,c.z=0,c.w=0,c.next=function(){var l=c.x^c.x<<11;return c.x=c.y,c.y=c.z,c.z=c.w,c.w^=c.w>>>19^l^l>>>8},i===(i|0)?c.x=i:u+=i;for(var h=0;h<u.length+64;h++)c.x^=u.charCodeAt(h)|0,c.next()}function o(i,c){return c.x=i.x,c.y=i.y,c.z=i.z,c.w=i.w,c}function a(i,c){var u=new r(i),h=c&&c.state,l=function(){return(u.next()>>>0)/4294967296};return l.double=function(){do var d=u.next()>>>11,g=(u.next()>>>0)/4294967296,w=(d+g)/(1<<21);while(w===0);return w},l.int32=u.next,l.quick=l,h&&(typeof h=="object"&&o(h,u),l.state=function(){return o(u,{})}),l}n&&n.exports?n.exports=a:s&&s.amd?s(function(){return a}):this.xor128=a})(oe,e,!1)})(fs);var Sp=fs.exports,ds={exports:{}};ds.exports;(function(e){(function(t,n,s){function r(i){var c=this,u="";c.next=function(){var l=c.x^c.x>>>2;return c.x=c.y,c.y=c.z,c.z=c.w,c.w=c.v,(c.d=c.d+362437|0)+(c.v=c.v^c.v<<4^(l^l<<1))|0},c.x=0,c.y=0,c.z=0,c.w=0,c.v=0,i===(i|0)?c.x=i:u+=i;for(var h=0;h<u.length+64;h++)c.x^=u.charCodeAt(h)|0,h==u.length&&(c.d=c.x<<10^c.x>>>4),c.next()}function o(i,c){return c.x=i.x,c.y=i.y,c.z=i.z,c.w=i.w,c.v=i.v,c.d=i.d,c}function a(i,c){var u=new r(i),h=c&&c.state,l=function(){return(u.next()>>>0)/4294967296};return l.double=function(){do var d=u.next()>>>11,g=(u.next()>>>0)/4294967296,w=(d+g)/(1<<21);while(w===0);return w},l.int32=u.next,l.quick=l,h&&(typeof h=="object"&&o(h,u),l.state=function(){return o(u,{})}),l}n&&n.exports?n.exports=a:s&&s.amd?s(function(){return a}):this.xorwow=a})(oe,e,!1)})(ds);var Tp=ds.exports,ps={exports:{}};ps.exports;(function(e){(function(t,n,s){function r(i){var c=this;c.next=function(){var h=c.x,l=c.i,d,g;return d=h[l],d^=d>>>7,g=d^d<<24,d=h[l+1&7],g^=d^d>>>10,d=h[l+3&7],g^=d^d>>>3,d=h[l+4&7],g^=d^d<<7,d=h[l+7&7],d=d^d<<13,g^=d^d<<9,h[l]=g,c.i=l+1&7,g};function u(h,l){var d,g=[];if(l===(l|0))g[0]=l;else for(l=""+l,d=0;d<l.length;++d)g[d&7]=g[d&7]<<15^l.charCodeAt(d)+g[d+1&7]<<13;for(;g.length<8;)g.push(0);for(d=0;d<8&&g[d]===0;++d);for(d==8?g[7]=-1:g[d],h.x=g,h.i=0,d=256;d>0;--d)h.next()}u(c,i)}function o(i,c){return c.x=i.x.slice(),c.i=i.i,c}function a(i,c){i==null&&(i=+new Date);var u=new r(i),h=c&&c.state,l=function(){return(u.next()>>>0)/4294967296};return l.double=function(){do var d=u.next()>>>11,g=(u.next()>>>0)/4294967296,w=(d+g)/(1<<21);while(w===0);return w},l.int32=u.next,l.quick=l,h&&(h.x&&o(h,u),l.state=function(){return o(u,{})}),l}n&&n.exports?n.exports=a:s&&s.amd?s(function(){return a}):this.xorshift7=a})(oe,e,!1)})(ps);var _p=ps.exports,gs={exports:{}};gs.exports;(function(e){(function(t,n,s){function r(i){var c=this;c.next=function(){var h=c.w,l=c.X,d=c.i,g,w;return c.w=h=h+1640531527|0,w=l[d+34&127],g=l[d=d+1&127],w^=w<<13,g^=g<<17,w^=w>>>15,g^=g>>>12,w=l[d]=w^g,c.i=d,w+(h^h>>>16)|0};function u(h,l){var d,g,w,y,$,k=[],I=128;for(l===(l|0)?(g=l,l=null):(l=l+"\0",g=0,I=Math.max(I,l.length)),w=0,y=-32;y<I;++y)l&&(g^=l.charCodeAt((y+32)%l.length)),y===0&&($=g),g^=g<<10,g^=g>>>15,g^=g<<4,g^=g>>>13,y>=0&&($=$+1640531527|0,d=k[y&127]^=g+$,w=d==0?w+1:0);for(w>=128&&(k[(l&&l.length||0)&127]=-1),w=127,y=4*128;y>0;--y)g=k[w+34&127],d=k[w=w+1&127],g^=g<<13,d^=d<<17,g^=g>>>15,d^=d>>>12,k[w]=g^d;h.w=$,h.X=k,h.i=w}u(c,i)}function o(i,c){return c.i=i.i,c.w=i.w,c.X=i.X.slice(),c}function a(i,c){i==null&&(i=+new Date);var u=new r(i),h=c&&c.state,l=function(){return(u.next()>>>0)/4294967296};return l.double=function(){do var d=u.next()>>>11,g=(u.next()>>>0)/4294967296,w=(d+g)/(1<<21);while(w===0);return w},l.int32=u.next,l.quick=l,h&&(h.X&&o(h,u),l.state=function(){return o(u,{})}),l}n&&n.exports?n.exports=a:s&&s.amd?s(function(){return a}):this.xor4096=a})(oe,e,!1)})(gs);var Ip=gs.exports,ms={exports:{}};ms.exports;(function(e){(function(t,n,s){function r(i){var c=this,u="";c.next=function(){var l=c.b,d=c.c,g=c.d,w=c.a;return l=l<<25^l>>>7^d,d=d-g|0,g=g<<24^g>>>8^w,w=w-l|0,c.b=l=l<<20^l>>>12^d,c.c=d=d-g|0,c.d=g<<16^d>>>16^w,c.a=w-l|0},c.a=0,c.b=0,c.c=-1640531527,c.d=1367130551,i===Math.floor(i)?(c.a=i/4294967296|0,c.b=i|0):u+=i;for(var h=0;h<u.length+20;h++)c.b^=u.charCodeAt(h)|0,c.next()}function o(i,c){return c.a=i.a,c.b=i.b,c.c=i.c,c.d=i.d,c}function a(i,c){var u=new r(i),h=c&&c.state,l=function(){return(u.next()>>>0)/4294967296};return l.double=function(){do var d=u.next()>>>11,g=(u.next()>>>0)/4294967296,w=(d+g)/(1<<21);while(w===0);return w},l.int32=u.next,l.quick=l,h&&(typeof h=="object"&&o(h,u),l.state=function(){return o(u,{})}),l}n&&n.exports?n.exports=a:s&&s.amd?s(function(){return a}):this.tychei=a})(oe,e,!1)})(ms);var Dp=ms.exports,xo={exports:{}};const Np={},Ap=Object.freeze(Object.defineProperty({__proto__:null,default:Np},Symbol.toStringTag,{value:"Module"})),Mp=iu(Ap);(function(e){(function(t,n,s){var r=256,o=6,a=52,i="random",c=s.pow(r,o),u=s.pow(2,a),h=u*2,l=r-1,d;function g(E,T,_){var A=[];T=T==!0?{entropy:!0}:T||{};var N=k($(T.entropy?[E,D(n)]:E??I(),3),A),F=new w(A),P=function(){for(var C=F.g(o),q=c,L=0;C<u;)C=(C+L)*r,q*=r,L=F.g(1);for(;C>=h;)C/=2,q/=2,L>>>=1;return(C+L)/q};return P.int32=function(){return F.g(4)|0},P.quick=function(){return F.g(4)/4294967296},P.double=P,k(D(F.S),n),(T.pass||_||function(C,q,L,V){return V&&(V.S&&y(V,F),C.state=function(){return y(F,{})}),L?(s[i]=C,q):C})(P,N,"global"in T?T.global:this==s,T.state)}function w(E){var T,_=E.length,A=this,N=0,F=A.i=A.j=0,P=A.S=[];for(_||(E=[_++]);N<r;)P[N]=N++;for(N=0;N<r;N++)P[N]=P[F=l&F+E[N%_]+(T=P[N])],P[F]=T;(A.g=function(C){for(var q,L=0,V=A.i,tt=A.j,wt=A.S;C--;)q=wt[V=l&V+1],L=L*r+wt[l&(wt[V]=wt[tt=l&tt+q])+(wt[tt]=q)];return A.i=V,A.j=tt,L})(r)}function y(E,T){return T.i=E.i,T.j=E.j,T.S=E.S.slice(),T}function $(E,T){var _=[],A=typeof E,N;if(T&&A=="object")for(N in E)try{_.push($(E[N],T-1))}catch{}return _.length?_:A=="string"?E:E+"\0"}function k(E,T){for(var _=E+"",A,N=0;N<_.length;)T[l&N]=l&(A^=T[l&N]*19)+_.charCodeAt(N++);return D(T)}function I(){try{var E;return d&&(E=d.randomBytes)?E=E(r):(E=new Uint8Array(r),(t.crypto||t.msCrypto).getRandomValues(E)),D(E)}catch{var T=t.navigator,_=T&&T.plugins;return[+new Date,t,_,t.screen,D(n)]}}function D(E){return String.fromCharCode.apply(0,E)}if(k(s.random(),n),e.exports){e.exports=g;try{d=Mp}catch{}}else s["seed"+i]=g})(typeof self<"u"?self:oe,[],Math)})(xo);var Fp=xo.exports,Bp=Ep,Cp=Sp,Rp=Tp,Pp=_p,Op=Ip,Lp=Dp,ie=Fp;ie.alea=Bp;ie.xor128=Cp;ie.xorwow=Rp;ie.xorshift7=Pp;ie.xor4096=Op;ie.tychei=Lp;var bs=ie;/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class ws{constructor(t,n,s,r,o){this.mean=t,this.stdDev=n,this.dtype=s,this.nextVal=NaN,this.truncated=r,this.truncated&&(this.upper=this.mean+this.stdDev*2,this.lower=this.mean-this.stdDev*2);const a=o||Math.random();this.random=bs.alea(a.toString())}nextValue(){if(!isNaN(this.nextVal)){const r=this.nextVal;return this.nextVal=NaN,r}let t,n,s=!1;for(;!s;){let r,o,a;do r=2*this.random()-1,o=2*this.random()-1,a=r*r+o*o;while(a>=1||a===0);const i=Math.sqrt(-2*Math.log(a)/a);t=this.mean+this.stdDev*r*i,n=this.mean+this.stdDev*o*i,(!this.truncated||this.isValidTruncated(t))&&(s=!0)}return(!this.truncated||this.isValidTruncated(n))&&(this.nextVal=this.convertValue(n)),this.convertValue(t)}convertValue(t){return this.dtype==null||this.dtype==="float32"?t:Math.round(t)}isValidTruncated(t){return t<=this.upper&&t>=this.lower}}class Up{constructor(t,n,s,r){this.alpha=t,this.beta=1/n,this.dtype=s;const o=r||Math.random();this.randu=bs.alea(o.toString()),this.randn=new ws(0,1,s,!1,this.randu()),t<1?this.d=t+2/3:this.d=t-1/3,this.c=1/Math.sqrt(9*this.d)}nextValue(){let t,n,s,r,o,a;for(;;){do r=this.randn.nextValue(),a=1+this.c*r;while(a<=0);if(a*=a*a,t=r*r,n=1-.331*t*t,s=.5*t+this.d*(1-a+Math.log(a)),o=this.randu(),o<n||Math.log(o)<s)break}return a=1/this.beta*this.d*a,this.alpha<1&&(a*=Math.pow(this.randu(),1/this.alpha)),this.convertValue(a)}convertValue(t){return this.dtype==="float32"?t:Math.round(t)}}class Wp{constructor(t=0,n=1,s,r){if(this.canReturnFloat=()=>this.dtype==null||this.dtype==="float32",this.min=t,this.range=n-t,this.dtype=s,r==null&&(r=Math.random()),typeof r=="number"&&(r=r.toString()),!this.canReturnFloat()&&this.range<=1)throw new Error(`The difference between ${t} - ${n} <= 1 and dtype is not float`);this.random=bs.alea(r)}convertValue(t){return this.canReturnFloat()?t:Math.round(t)}nextValue(){return this.convertValue(this.min+this.range*this.random())}}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function qp(e,t,n=1,s="float32",r){if(dt(e),n==null&&(n=1),s==null&&(s="float32"),s!=="float32"&&s!=="int32")throw new Error(`Unsupported data type ${s}`);const o=new Up(t,n,s,r),a=At(e,s);for(let i=0;i<a.values.length;i++)a.values[i]=o.nextValue();return a.toTensor()}const Kp=m({randomGamma_:qp});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function zp(e,t=0,n=1,s,r){if(dt(e),s!=null&&s==="bool")throw new Error(`Unsupported data type ${s}`);const o=new ws(t,n,s,!1,r),a=At(e,s);for(let i=0;i<a.values.length;i++)a.values[i]=o.nextValue();return a.toTensor()}const ko=m({randomNormal_:zp});/**
 * @license
 * Copyright 2022 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Gp(e,t,n){if(t!=null&&t==="bool")throw new Error(`Unsupported data type ${t}`);return ko(e,0,1,t,n)}const Hp=m({randomStandardNormal_:Gp});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Vp(e,t=0,n=1,s="float32",r){dt(e);const o=At(e,s),a=new Wp(t,n,null,r);for(let i=0;i<o.values.length;i++)o.values[i]=a.nextValue();return o.toTensor()}const ys=m({randomUniform_:Vp});/**
 * @license
 * Copyright 2023 Google LLC.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function jp(e,t,n,s){return ys(e,t,n,"int32",s)}const Xp=m({randomUniformInt_:jp});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function We(e,t,n=1,s="float32"){if(n===0)throw new Error("Cannot have a step of zero");const r={start:e,stop:t,step:n,dtype:s};return b.runKernel(lc,{},r)}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Yp(e){const n={input:f(e,"input","real")};return b.runKernel(hc,n)}const qe=m({real_:Yp});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Zp(e){const n={x:f(e,"x","reciprocal")};return b.runKernel(fc,n)}const Jp=m({reciprocal_:Zp});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Qp(e){const n={x:f(e,"x","relu")};return b.runKernel(dc,n)}const Tn=m({relu_:Qp});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function tg(e){const n={x:f(e,"x","relu6")};return b.runKernel(bc,n)}const vo=m({relu6_:tg});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function eg(e,t){const s={x:f(e,"x","reverse")},r={dims:t};return b.runKernel(wc,s,r)}const ne=m({reverse_:eg});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function ng(e){const t=f(e,"x","reverse");return p(t.rank===1,()=>`Error in reverse1D: x must be rank 1 but got rank ${t.rank}.`),ne(t,0)}const sg=m({reverse1d_:ng});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function rg(e,t){const n=f(e,"x","reverse");return p(n.rank===2,()=>`Error in reverse2D: x must be rank 2 but got rank ${n.rank}.`),ne(n,t)}const og=m({reverse2d_:rg});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function ag(e,t){const n=f(e,"x","reverse");return p(n.rank===3,()=>`Error in reverse3D: x must be rank 3 but got rank ${n.rank}.`),ne(n,t)}const ig=m({reverse3d_:ag});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function cg(e,t){const n=f(e,"x","reverse");return p(n.rank===4,()=>`Error in reverse4D: x must be rank 4 but got rank ${n.rank}.`),ne(n,t)}const ug=m({reverse4d_:cg});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function lg(e){const n={x:f(e,"x","round")};return b.runKernel(yc,n)}const Eo=m({round_:lg});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function hg(e){const n={x:f(e,"x","rsqrt","float32")};return b.runKernel($c,n)}const fg=m({rsqrt_:hg});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function dg(e){const n={x:f(e,"x","selu")};return b.runKernel(Sc,n)}const pg=m({selu_:dg});function gg(e,t,n,s,r,o=[1,1],a="NHWC"){const i=f(e,"x","separableConv2d"),c=f(t,"depthwiseFilter","separableConv2d"),u=f(n,"pointwiseFilter","separableConv2d");let h=i,l=!1;if(i.rank===3&&(l=!0,h=x(i,[1,i.shape[0],i.shape[1],i.shape[2]])),a==="NCHW")throw new Error("separableConv2d currently does not support dataFormat NCHW; only NHWC is supported");p(h.rank===4,()=>`Error in separableConv2d: input must be rank 4, but got rank ${h.rank}.`),p(c.rank===4,()=>`Error in separableConv2d: depthwise filter must be rank 4, but got rank ${c.rank}.`),p(u.rank===4,()=>`Error in separableConv2d: pointwise filter must be rank 4, but got rank ${c.rank}.`),p(u.shape[0]===1,()=>`Error in separableConv2d: the first dimension of pointwise filter  must be 1, but got ${u.shape[0]}.`),p(u.shape[1]===1,()=>`Error in separableConv2d: the second dimension of pointwise filter must be 1, but got ${u.shape[1]}.`);const d=c.shape[2],g=c.shape[3];p(u.shape[2]===d*g,()=>`Error in separableConv2d: the third dimension of pointwise filter must be ${d*g}, but got ${u.shape[2]}.`);const w=cs(h,c,s,r,a,o),$=$n(w,u,1,"valid",a);return l?x($,[$.shape[1],$.shape[2],$.shape[3]]):$}const mg=m({separableConv2d_:gg});/**
 * @license
 * Copyright 2020 Google Inc. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */async function bg(e,t){const n=f(e,"x","setdiff1d"),s=f(t,"y","setdiff1d");p(n.dtype===s.dtype,()=>`x and y should have the same dtype, but got x (${n.dtype}) and y (${s.dtype}).`),p(n.rank===1,()=>`x should be 1D tensor, but got x (${n.shape}).`),p(s.rank===1,()=>`y should be 1D tensor, but got y (${s.shape}).`);const r=await n.data(),o=await s.data(),a=new Set(o);let i=0;for(let h=0;h<r.length;h++)a.has(r[h])||i++;const c=new Pn([i],n.dtype),u=new Pn([i],"int32");for(let h=0,l=0;h<r.length;h++)a.has(r[h])||(c.values[l]=r[h],u.values[l]=h,l++);return[c.toTensor(),u.toTensor()]}const wg=bg;/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function yg(e){const n={x:f(e,"x","sign")};return b.runKernel(Dc,n)}const $g=m({sign_:yg});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function xg(e){const n={x:f(e,"x","sin","float32")};return b.runKernel(_c,n)}const kg=m({sin_:xg});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function vg(e){const n={x:f(e,"x","sinh")};return b.runKernel(Ic,n)}const Eg=m({sinh_:vg});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Sg(e,t,n){const s=f(e,"x","slice1d");return p(s.rank===1,()=>`slice1d expects a rank-1 tensor, but got a rank-${s.rank} tensor`),j(s,[t],[n])}const Tg=m({slice1d_:Sg});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function _g(e,t,n){const s=f(e,"x","slice2d");return p(s.rank===2,()=>`slice2d expects a rank-2 tensor, but got a rank-${s.rank} tensor`),j(s,t,n)}const Ig=m({slice2d_:_g});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Dg(e,t,n){const s=f(e,"x","slice3d");return p(s.rank===3,()=>`slice3d expects a rank-3 tensor, but got a rank-${s.rank} tensor`),j(s,t,n)}const Ng=m({slice3d_:Dg});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Ag(e,t,n){const s=f(e,"x","slice4d");return p(s.rank===4,()=>`slice4d expects a rank-4 tensor, but got a rank-${s.rank} tensor`),j(s,t,n)}const Mg=m({slice4d_:Ag});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Fg(e,t=-1){const n=f(e,"logits","softmax","float32");if(t===-1&&(t=n.rank-1),t!==n.rank-1)throw Error(`Softmax along a non-last dimension is not yet supported. Logits was rank ${n.rank} and dim was ${t}`);const s={logits:n},r={dim:t};return b.runKernel(Rc,s,r)}const Bg=m({softmax_:Fg});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Cg(e){p(e.dtype==="complex64",()=>`The dtype for tf.spectral.fft() must be complex64 but got ${e.dtype}.`);const t={input:e};return b.runKernel(fi,t)}const $s=m({fft_:Cg});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Rg(e){p(e.dtype==="complex64",()=>`The dtype for tf.spectral.ifft() must be complex64 but got ${e.dtype}.`);const t={input:e};return b.runKernel(ki,t)}const dn=m({ifft_:Rg});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Pg(e){const t=e.shape[e.shape.length-1],n=e.size/t;let s;if(t<=2){const r=x(e,[n,t]);s=dn(r)}else{const r=[n,2*(t-1)],o=x(qe(e),[n,t]),a=x(En(e),[n,t]),i=ne(j(o,[0,1],[n,t-2]),1),c=S(ne(j(a,[0,1],[n,t-2]),1),U(-1)),u=ft([o,i],1),h=ft([a,c],1),l=x(Ut(u,h),[r[0],r[1]]);s=dn(l)}if(s=qe(s),e.rank===3&&e.shape[0]!==0){const r=s,o=e.shape[0];s=x(s,[o,s.shape[0]/o,s.shape[1]]),r.dispose()}return s}const So=m({irfft_:Pg});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Og(e,t,n=0){const r={x:f(e,"x","split")},o={numOrSizeSplits:t,axis:n};return b.runKernel(Cc,r,o)}const Ke=m({split_:Og});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Lg(e,t){p(e.dtype==="float32",()=>`The dtype for rfft() must be real value but got ${e.dtype}`);let n=e.shape[e.shape.length-1];const s=e.size/n;let r;if(t!=null&&t<n){const w=e.shape.map($=>0),y=e.shape.map($=>$);y[e.shape.length-1]=t,r=j(e,w,y),n=t}else if(t!=null&&t>n){const w=e.shape.map(y=>y);w[e.shape.length-1]=t-n,r=ft([e,Te(w)],e.shape.length-1),n=t}else r=e;const o=mt(r),a=x(Ut(r,o),[s,n]),i=$s(a),c=Math.floor(n/2)+1,u=qe(i),h=En(i),l=Ke(u,[c,n-c],u.shape.length-1),d=Ke(h,[c,n-c],h.shape.length-1),g=r.shape.slice();return g[r.shape.length-1]=c,x(Ut(l[0],d[0]),g)}const xs=m({rfft_:Lg});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Ug(e,t){let n=f(e,"a","squaredDifference"),s=f(t,"b","squaredDifference");[n,s]=Y(n,s),Q(n.shape,s.shape);const r={a:n,b:s},o={};return b.runKernel(qc,r,o)}const To=m({squaredDifference_:Ug});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Wg(e,t){const n=f(e,"x","squeeze","string_or_numeric");return x(n,sr(n.shape,t).newShape)}const ks=m({squeeze_:Wg});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function qg(e,t=0){const n=Re(e,"tensors","stack","string_or_numeric");p(n.length>=1,()=>"Pass at least one tensor to tf.stack"),n.length>0&&p(t<=n[0].rank,()=>"Axis must be <= rank of the tensor");const s=n,r={axis:t};return b.runKernel(nc,s,r)}const ze=m({stack_:qg});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Kg(e,t=0){const s={x:f(e,"x","step")},r={alpha:t};return b.runKernel(su,s,r)}const _o=m({step_:Kg});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function zg(e,t,n,s,r=0,o=0,a=0,i=0,c=0){const h={x:f(e,"x","stridedSlice","string_or_numeric")},l={begin:t,end:n,strides:s,beginMask:r,endMask:o,ellipsisMask:a,newAxisMask:i,shrinkAxisMask:c};return b.runKernel(zc,h,l)}const Gg=m({stridedSlice_:zg});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Hg(e){const n={x:f(e,"x","tan","float32")};return b.runKernel(Xc,n)}const Vg=m({tan_:Hg});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function vt(e,t){re(e);const n=qt(e,t);if(n.length!==1)throw new Error("tensor1d() requires values to be a flat/TypedArray");return Kt(e,null,n,t)}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Fe(e,t,n){if(re(e),t!=null&&t.length!==2)throw new Error("tensor2d() requires shape to have two numbers");const s=qt(e,n);if(s.length!==2&&s.length!==1)throw new Error("tensor2d() requires values to be number[][] or flat/TypedArray");if(s.length===1&&t==null)throw new Error("tensor2d() requires shape to be provided when `values` are a flat/TypedArray");return Kt(e,t,s,n)}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function jg(e,t,n){if(re(e),t!=null&&t.length!==3)throw new Error("tensor3d() requires shape to have three numbers");const s=qt(e,n);if(s.length!==3&&s.length!==1)throw new Error("tensor3d() requires values to be number[][][] or flat/TypedArray");if(s.length===1&&t==null)throw new Error("tensor3d() requires shape to be provided when `values` are a flat array");return Kt(e,t,s,n)}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Xg(e,t,n){if(re(e),t!=null&&t.length!==4)throw new Error("tensor4d() requires shape to have four numbers");const s=qt(e,n);if(s.length!==4&&s.length!==1)throw new Error("tensor4d() requires values to be number[][][][] or flat/TypedArray");if(s.length===1&&t==null)throw new Error("tensor4d() requires shape to be provided when `values` are a flat array");return Kt(e,t,s,n)}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Yg(e,t,n){if(re(e),t!=null&&t.length!==5)throw new Error("tensor5d() requires shape to have five numbers");const s=qt(e,n);if(s.length!==5&&s.length!==1)throw new Error("tensor5d() requires values to be number[][][][][] or flat/TypedArray");if(s.length===1&&t==null)throw new Error("tensor5d() requires shape to be provided when `values` are a flat array");return Kt(e,t,s,n)}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Zg(e,t,n){if(re(e),t!=null&&t.length!==6)throw new Error("tensor6d() requires shape to have six numbers");const s=qt(e,n);if(s.length!==6&&s.length!==1)throw new Error("tensor6d() requires values to be number[][][][][][] or flat/TypedArray");if(s.length===1&&t==null)throw new Error("tensor6d() requires shape to be provided when `values` are a flat array");return t=t||s,Kt(e,t,s,n)}function Io(e,t,n){const s=t.rank>1?t.shape[t.rank-1]:1,r=t.rank>1?t.rank-1:1,o=`Must have updates.shape = indices.shape[:batchDim] + shape[sliceDim:], got updates.shape: ${n.shape}, indices.shape: ${t.shape}, shape: ${e}, sliceDim: ${s}, and batchDim: ${r}.`;if(n.rank<r)throw new Error(o+` update.rank < ${r}. `);if(e.length<s+(n.rank-r))throw new Error(o+` Output shape length < ${s+(n.rank-r)}`);if(n.rank!==r+e.length-s)throw new Error(o+` update.rank != ${r+e.length-s}`);for(let a=0;a<r;++a)if(n.shape[a]!==t.shape[a])throw new Error(o+` updates.shape[${a}] (${n.shape[a]}) != indices.shape[${a}] (${t.shape[a]}).`);for(let a=0;a<n.rank-r;++a)if(n.shape[a+r]!==e[a+s])throw new Error(o+` updates.shape[${a+r}] (${n.shape[a+r]}) != shape[${a+r}] (${e[a+r]})`)}function vs(e,t,n){if(t.rank<1)throw new Error(`tf.scatterND() expects the indices to be rank 1 or higher, but the rank was ${t.rank}.`);if(e.rank<1)throw new Error(`tf.scatterND() expects the updates to be rank 1 or higher, but the rank was ${e.rank}.`);if(t.dtype!=="int32")throw new Error(`The dtype of 'indices' should be int32, but got dtype: ${t.dtype}`);if(n.length<1)throw new Error(`Output rank must be greater or equal to 1, but got shape: ${n}`);if(n.length===0){if(t.size===0)throw new Error(`Indices specified for empty output. indices shape: ${t.shape}`);if(e.size===0)throw new Error(`Updates specified for empty output. updates shape: ${e.shape}`)}Io(n,t,e)}function Jg(e,t,n){const s=t.shape.length,r=s>1?t.shape[s-1]:1,o=n.length;let a=1;for(let l=r;l<o;++l)a*=n[l];const i=r<1?1:r,c=Z(t.shape)/i,u=[...Ve(n.slice(0,r)),1],h=Z(n);return{sliceRank:r,numUpdates:c,sliceSize:a,strides:u,outputSize:h}}const o0=Object.freeze(Object.defineProperty({__proto__:null,calculateShapes:Jg,validateInput:vs,validateUpdateShape:Io},Symbol.toStringTag,{value:"Module"}));/**
 * @license
 * Copyright 2022 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Qg(e,t,n){const s=f(e,"tensor","tensorScatterupdate"),r=f(t,"indices","tensorScatterupdate","int32"),o=f(n,"updates","tensorScatterupdate");if(vs(o,r,s.shape),s.dtype!==o.dtype)throw new Error(`tensor and updates must have the same dtype, instead they are ${s.dtype} and ${o.dtype}.`);const a={tensor:s,indices:r,updates:o},i={};return b.runKernel(kc,a,i)}const tm=m({tensorScatterUpdate_:Qg});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function em(e,t=1,n=!0){const s=f(e,"x","topk");if(s.rank===0)throw new Error("topk() expects the input to be of rank 1 or higher");const r=s.shape[s.shape.length-1];if(t<0)throw new Error(`'k' passed to topk() must be >= 0 but got ${t}`);if(t>r)throw new Error(`'k' passed to topk() must be <= the last dimension (${r}) but got ${t}`);const o={x:s},a={k:t,sorted:n},[i,c]=b.runKernel(Zc,o,a);return{values:i,indices:c}}const nm=m({topk_:em});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function sm(e,t=0,n=1,s,r){if(dt(e),s!=null&&s==="bool")throw new Error("Unsupported data type $ { dtype }");const o=new ws(t,n,s,!0,r),a=At(e,s);for(let i=0;i<a.values.length;i++)a.values[i]=o.nextValue();return a.toTensor()}const rm=m({truncatedNormal_:sm});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function om(e,t=0){const n=f(e,"x","unique","string_or_numeric");p(n.rank>0,()=>"The input tensor must be at least 1D");const s={x:n},r={axis:t},[o,a]=b.runKernel(Qc,s,r);return{values:o,indices:a}}const am=m({unique_:om});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function im(e,t,n){const s=f(e,"x","unsortedSegmentSum"),r=f(t,"segmentIds","unsortedSegmentSum","int32");p($e(n),()=>"numSegments must be of dtype int");const o={x:s,segmentIds:r},a={numSegments:n};return b.runKernel(eu,o,a)}const cm=m({unsortedSegmentSum_:im});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function um(e,t=0){const n=f(e,"x","unstack","string_or_numeric");p(t>=-n.shape.length&&t<n.shape.length,()=>`Axis = ${t} is not in [-${n.shape.length}, ${n.shape.length})`);const s={value:n},r={axis:t};return b.runKernel(tu,s,r)}const Es=m({unstack_:um});/**
 * @license
 * Copyright 2022 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function lm(e,t){return ls(e,t,"right")}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function hm(e,t=!0,n,s){return b.makeVariable(e,t,n,s)}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function fm(e,t){const n=[];for(let o=0;o<t.length;o++)t[o]&&n.push(o);const s=At(e,"int32"),r=At([n.length,e.length],"int32");for(let o=0;o<n.length;o++){const a=s.indexToLoc(n[o]),i=o*e.length;r.values.set(a,i)}return r.toTensor()}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */async function dm(e){const t=f(e,"condition","whereAsync","bool"),n=await t.data(),s=fm(t.shape,n);return e!==t&&t.dispose(),s}const Do=dm;/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */async function pm(e,t,n){const s=f(e,"tensor","boolMask"),r=f(t,"mask","boolMask","bool"),o=n??0,a=r.rank,i=s.shape;p(a>0,()=>"mask cannot be scalar"),at(i.slice(o,o+a),r.shape,"mask's shape must match the first K dimensions of tensor's shape,");let c=1;for(let y=o;y<o+a;y++)c*=i[y];const u=i.slice(0,o).concat([c],i.slice(o+a)),h=x(s,u),l=x(r,[-1]),d=await Do(l),g=ks(d,[1]),w=io(h,g,o);return e!==s&&s.dispose(),t!==r&&r.dispose(),g.dispose(),h.dispose(),l.dispose(),d.dispose(),w}const gm=pm;/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function mm(e,t,n){const s=f(e,"x","transpose");if(t==null&&(t=s.shape.map((a,i)=>i).reverse()),p(s.rank===t.length,()=>`Error in transpose: rank of input ${s.rank} must match length of perm ${t}.`),t.forEach(a=>{p(a>=0&&a<s.rank,()=>`All entries in 'perm' must be between 0 and ${s.rank-1} but got ${t}`)}),s.rank<=1)return s.clone();const r={x:s},o={perm:t};return s.dtype==="complex64"?J(()=>{let a=qe(s),i=En(s);return a=b.runKernel(In,{x:a},o),i=b.runKernel(In,{x:i},o),n&&(i=_t(i)),Ut(a,i)}):b.runKernel(In,r,o)}const Zn=m({transpose_:mm});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function bm(e,t,n,s,r=!0){const o=f(e,"v","movingAverage"),a=f(t,"x","movingAverage"),i=f(n,"decay","movingAverage");Nr(o,a),p(Wt(o.shape,a.shape),()=>"Shape mismatch in v and x");const c=U(1),u=R(c,i);let h=S(R(a,o),u);if(r){p(s!=null,()=>"When using zeroDebias: true, step is required.");const l=f(s,"step","movingAverage");h=z(h,R(c,Le(i,l)))}return M(o,h)}const wm=m({movingAverage_:bm});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function ym(e,t,n){dt(n);const s=f(e,"indices","scatterND","int32"),r=f(t,"updates","scatterND");vs(r,s,n);const o={indices:s,updates:r},a={shape:n};return b.runKernel(xc,o,a)}const $m=m({scatterND_:ym});function xm(e,t,n,s){if(e.dtype!=="int32")throw new Error(`tf.sparseToDense() expects the indices to be int32 type, but the dtype was ${e.dtype}.`);if(e.rank>2)throw new Error(`sparseIndices should be a scalar, vector, or matrix, but got shape ${e.shape}.`);const r=e.rank>0?e.shape[0]:1,o=e.rank>1?e.shape[1]:1;if(n.length!==o)throw new Error(`outputShape has incorrect number of elements:, ${n.length}, should be: ${o}.`);const a=t.size;if(!(t.rank===0||t.rank===1&&a===r))throw new Error(`sparseValues has incorrect shape ${t.shape}, should be [] or [${r}]`);if(t.dtype!==s.dtype)throw new Error("sparseValues.dtype must match defaultValues.dtype")}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function km(e,t,n,s=0){dt(n);const r=f(e,"sparseIndices","sparseToDense","int32"),o=f(t,"sparseValues","sparseToDense","string_or_numeric"),a=f(s,"defaultValue","sparseToDense",o.dtype);xm(r,o,n,a);const i={sparseIndices:r,sparseValues:o,defaultValue:a},c={outputShape:n};return b.runKernel(Wc,i,c)}const vm=m({sparseToDense_:km});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Em(e,t){const n=f(t,"indices","gatherND","int32"),r={params:f(e,"x","gatherND","string_or_numeric"),indices:n};return b.runKernel(yi,r)}const Sm=m({gatherND_:Em});/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Tm(e,t){if(t==null)return e.shape.slice();if(Wt(e.shape,t))return t;if(e.shape.length===t.length){const n=[];for(let s=0;s<e.shape.length;s++)t[s]==null&&e.shape[s]!=null?n.push(e.shape[s]):n.push(t[s]);return n}return t}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function _m(e,t,n,s){const r=f(e,"x","dropout");if(p(r.dtype==="float32",()=>`x has to be a floating point tensor since it's going to be scaled, but got a ${r.dtype} tensor instead.`),p(t>=0&&t<1,()=>`rate must be a float in the range [0, 1), but got ${t}.`),t===0)return e instanceof nt?r.clone():r;const o=Tm(r,n),a=1-t,i=z(ao(M(ys(o,0,1,"float32",s),a)),a);return S(r,i)}const Im=m({dropout_:_m});/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function No(e){return Math.floor(Math.pow(2,Math.ceil(Math.log(e)/Math.log(2))))}function Ss(e,t,n){const s=1-e%2,r=new Float32Array(e);for(let o=0;o<e;++o){const a=2*Math.PI*o/(e+s-1);r[o]=t-n*Math.cos(a)}return vt(r,"float32")}/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */async function Dm(e,t,n=1){const s=f(e,"predictions","inTopK"),r=f(t,"targets","inTopK");p(s.rank>1,()=>`inTopK() expects the predictions to be of rank 2 or higher, but got ${s.rank}`),p(s.rank-1===r.rank,()=>`predictions rank should be 1 larger than targets rank, but got predictions rank ${s.rank} and targets rank ${r.rank}`),at(s.shape.slice(0,s.shape.length-1),r.shape,"predictions's shape should be align with the targets' shape, except the last dimension.");const o=s.shape[s.shape.length-1];p(n>0&&n<=o,()=>`'k' passed to inTopK() must be > 0 && <= the predictions last dimension (${o}), but got ${n}`);const a=await s.data(),i=await r.data(),[c,u]=[a.length/o,o],h=rr("bool",c);for(let l=0;l<c;l++){const d=l*u,g=a.subarray(d,d+u),w=[];for(let y=0;y<g.length;y++)w.push({value:g[y],index:y});w.sort((y,$)=>$.value-y.value),h[l]=0;for(let y=0;y<n;y++)if(w[y].index===i[l]){h[l]=1;break}}return e!==s&&s.dispose(),t!==r&&r.dispose(),ge(h,r.shape,"bool")}const Nm=Dm;/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Am(e,t,n,s,r,o="NHWC",a){let i=e;e.rank===3&&(i=x(e,[1,e.shape[0],e.shape[1],e.shape[2]]));let c=t;c.rank===3&&(c=x(t,[1,t.shape[0],t.shape[1],t.shape[2]])),p(i.rank===4,()=>`Error in conv2dDerFilter: input must be rank 4, but got shape ${i.shape}.`),p(c.rank===4,()=>`Error in conv2dDerFilter: dy must be rank 4, but got shape ${c.shape}.`),p(n.length===4,()=>`Error in conv2dDerFilter: filterShape must be length 4, but got ${n}.`);const u=o==="NHWC"?i.shape[3]:i.shape[1],h=o==="NHWC"?c.shape[3]:c.shape[1];p(u===n[2],()=>`Error in conv2dDerFilter: depth of input ${u}) must match input depth in filter (${n[2]}.`),p(h===n[3],()=>`Error in conv2dDerFilter: depth of dy (${h}) must match output depth for filter (${n[3]}).`),Tt("conv2dDerFilter",r,a);const l={x:i,dy:c},d={strides:s,pad:r,dataFormat:o,dimRoundingMode:a,filterShape:n};return b.runKernel(qa,l,d)}const Mm=m({conv2DBackpropFilter_:Am});/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Ts(e,t,n){if(n==null||n==="linear")return e;if(n==="relu")return S(e,_o(t));throw new Error(`Cannot compute gradient for fused activation ${n}.`)}function _s(e,t){let n=t;const s=to(e.shape,t.shape);return s.length>0&&(n=K(n,s)),x(n,e.shape)}function Is(e,t,n,s){if(t==="linear")return e;if(t==="relu")return Tn(e);if(t==="elu")return no(e);if(t==="relu6")return vo(e);if(t==="prelu")return $o(e,n);if(t==="leakyrelu")return uo(e,s);if(t==="sigmoid")return we(e);throw new Error(`Unknown fused activation ${t}.`)}const Ds=(e,t)=>!(e>0)||t==="linear";/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Fm({x:e,filter:t,strides:n,pad:s,dataFormat:r="NHWC",dilations:o=[1,1],dimRoundingMode:a,bias:i,activation:c="linear",preluActivationWeights:u,leakyreluAlpha:h}){if(c=c||"linear",Ds(b.state.gradientDepth,c)===!1){p(r==="NHWC",()=>`Error in fused conv2d: got dataFormat of ${r} but only NHWC is currently supported for the case of gradient depth is 0 and the activation is not linear.`);let _=$n(e,t,n,s,r,o,a);return i!=null&&(_=M(_,i)),Is(_,c,u,h)}const l=f(e,"x","conv2d","float32"),d=f(t,"filter","conv2d","float32");let g=l,w=!1;l.rank===3&&(w=!0,g=x(l,[1,l.shape[0],l.shape[1],l.shape[2]])),p(g.rank===4,()=>`Error in fused conv2d: input must be rank 4, but got rank ${g.rank}.`),p(d.rank===4,()=>`Error in fused conv2d: filter must be rank 4, but got rank ${d.rank}.`),Tt("fused conv2d",s,a);const y=r==="NHWC"?g.shape[3]:g.shape[1];p(d.shape[2]===y,()=>`Error in conv2d: depth of input (${y}) must match input depth for filter ${d.shape[2]}.`),p(zt(n,o),()=>`Error in conv2D: Either strides or dilations must be 1. Got strides ${n} and dilations '${o}'`);const $=wn(g.shape,d.shape,n,o,s,a);let k;i!=null&&(k=f(i,"bias","fused conv2d"),[k]=Y(k,l),r==="NHWC"?Q($.outShape,k.shape):(p(k.shape.length<=1,()=>`Error in fused conv2d: only supports scalar or 1-D Tensor bias for NCHW format but got the bias of rank-${k.shape.length}.`),p(k.shape.length===0||k.shape[0]===$.outChannels||k.shape[0]===1,()=>`Error in fused conv2d: bias shape (${k.shape}) is not compatible with the number of output channels (${$.outChannels})`)));let I;if(u!=null){const _=u.shape;if(p(_.length<=1||_.length===3,()=>`Error in fused conv2d: only supports scalar, 1-D Tensor or 3-D Tensor PReLU activation weights but got a tensor of rank-${_.length}.`),_.length===1)p(_[0]===1||_[0]===$.outChannels,()=>`Error in fused conv2d: PReLU activation weights (${_}) is not compatible with the number of output channels (${$.outChannels}).`);else if(_.length===3)try{Q(_,$.outShape)}catch{const N=`Error in fused conv2d: PReLU activation weights (${_}) is not compatible with the output shape of the conv2d (${$.outShape}).`;throw Error(N)}I=f(u,"prelu weights","fused conv2d")}const D=(_,A)=>{p(r==="NHWC",()=>`Error in gradient of fused conv2D: got dataFormat of ${r} but only NHWC is currently supported.`);const[N,F,P,C]=A,q=Ts(_,P,c);p(un(o),()=>`Error in gradient of fused conv2D: dilation rates greater than 1 are not yet supported in gradients. Got dilations '${o}'`);const L=Qr(F.shape,q,N,n,s),V=Mm(F,q,N.shape,n,s),tt=[L,V];if(C!=null){const wt=_s(C,q);tt.push(wt)}return tt},E={x:g,filter:d,bias:k,preluActivationWeights:I},T={strides:n,pad:s,dataFormat:r,dilations:o,dimRoundingMode:a,activation:c,leakyreluAlpha:h};return i==null?Ft((A,N,F)=>{let P=b.runKernel(Cs,E,T);return F([N,A,P]),w&&(P=x(P,[P.shape[1],P.shape[2],P.shape[3]])),{value:P,gradFunc:D}})(g,d):Ft((A,N,F,P)=>{let C=b.runKernel(Cs,E,T);return P([N,A,C,F]),w&&(C=x(C,[C.shape[1],C.shape[2],C.shape[3]])),{value:C,gradFunc:D}})(g,d,k)}const Bm=m({fusedConv2d_:Fm});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Cm(e,t,n,s,r,o=[1,1],a){let i=e;e.rank===3&&(i=x(e,[1,e.shape[0],e.shape[1],e.shape[2]]));let c=t;c.rank===3&&(c=x(t,[1,t.shape[0],t.shape[1],t.shape[2]]));const u={x:i,dy:c},h={strides:s,pad:r,dimRoundingMode:a,dilations:o,filterShape:n};return b.runKernel(ti,u,h)}const Rm=m({depthwiseConv2dNativeBackpropFilter_:Cm});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Pm(e,t,n,s,r,o=[1,1],a){let i=t,c=!1;t.rank===3&&(c=!0,i=x(t,[1,t.shape[0],t.shape[1],t.shape[2]]));const u={dy:i,filter:n},h={strides:s,pad:r,dimRoundingMode:a,dilations:o,inputShape:e},l=b.runKernel(ei,u,h);return c?x(l,[l.shape[1],l.shape[2],l.shape[3]]):l}const Om=m({depthwiseConv2dNativeBackpropInput_:Pm});/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Lm({x:e,filter:t,strides:n,pad:s,dataFormat:r="NHWC",dilations:o=[1,1],dimRoundingMode:a,bias:i,activation:c="linear",preluActivationWeights:u,leakyreluAlpha:h}){if(Ds(b.state.gradientDepth,c)===!1){let T=cs(e,t,n,s,r,o,a);return i!=null&&(T=M(T,i)),Is(T,c,u,h)}const l=f(e,"x","depthwiseConv2d","float32"),d=f(t,"filter","depthwiseConv2d","float32");let g=l,w=!1;l.rank===3&&(w=!0,g=x(l,[1,l.shape[0],l.shape[1],l.shape[2]])),p(g.rank===4,()=>`Error in fused depthwiseConv2d: input must be rank 4, but got rank ${g.rank}.`),p(d.rank===4,()=>`Error in fused depthwiseConv2d: filter must be rank 4, but got rank ${d.rank}.`),p(g.shape[3]===d.shape[2],()=>`Error in fused depthwiseConv2d: number of input channels (${g.shape[3]}) must match the inChannels dimension in filter ${d.shape[2]}.`),o==null&&(o=[1,1]),p(zt(n,o),()=>`Error in fused depthwiseConv2d: Either strides or dilations must be 1. Got strides ${n} and dilations '${o}'`),Tt("fused depthwiseConv2d",s,a);const y=wn(g.shape,d.shape,n,o,s,a,!0);let $;i!=null&&($=f(i,"bias","fused conv2d"),[$]=Y($,l),Q(y.outShape,$.shape));let k;u!=null&&(k=f(u,"prelu weights","fused depthwiseConv2d"));const I=(T,_)=>{p(un(o),()=>`Error in gradient of fused depthwiseConv2d: dilation rates greater than 1 are not yet supported. Got dilations '${o}'`);const[A,N,F,P]=_,C=Ts(T,F,c),q=Om(N.shape,C,A,n,s,o,a),L=Rm(N,C,A.shape,n,s,o,a);if(P!=null){const V=_s($,C);return[q,L,V]}return[q,L]},D={x:g,filter:d,bias:$,preluActivationWeights:k},E={strides:n,pad:s,dataFormat:r,dilations:o,dimRoundingMode:a,activation:c,leakyreluAlpha:h};return i==null?Ft((_,A,N)=>{let F=b.runKernel(Rs,D,E);return N([A,_,F]),w&&(F=x(F,[F.shape[1],F.shape[2],F.shape[3]])),{value:F,gradFunc:I}})(g,d):Ft((_,A,N,F)=>{let P=b.runKernel(Rs,D,E);return F([A,_,P,N]),w&&(P=x(P,[P.shape[1],P.shape[2],P.shape[3]])),{value:P,gradFunc:I}})(g,d,$)}const Um=m({fusedDepthwiseConv2d_:Lm});/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Wm({a:e,b:t,transposeA:n=!1,transposeB:s=!1,bias:r,activation:o="linear",preluActivationWeights:a,leakyreluAlpha:i=.2}){if(Ds(b.state.gradientDepth,o)===!1){let C=O(e,t,n,s);return r!=null&&(C=M(C,r)),Is(C,o,a,i)}let c=f(e,"a","fused matMul"),u=f(t,"b","fused matMul");[c,u]=Y(c,u);const h=n?c.shape[c.rank-2]:c.shape[c.rank-1],l=s?u.shape[u.rank-1]:u.shape[u.rank-2],d=n?c.shape[c.rank-1]:c.shape[c.rank-2],g=s?u.shape[u.rank-2]:u.shape[u.rank-1],w=c.shape.slice(0,-2),y=u.shape.slice(0,-2),$=Z(w),k=Z(y);p(h===l,()=>`Error in fused matMul: inner shapes (${h}) and (${l}) of Tensors with shapes ${c.shape} and ${u.shape} and transposeA=${n} and transposeB=${s} must match.`);const D=Q(c.shape.slice(0,-2),u.shape.slice(0,-2)).concat([d,g]),E=n?x(c,[$,h,d]):x(c,[$,d,h]),T=s?x(u,[k,g,l]):x(u,[k,l,g]);let _;r!=null&&(_=f(r,"bias","fused matMul"),[_]=Y(_,c),Q(D,_.shape));let A;a!=null&&(A=f(a,"prelu weights","fused matMul"));const N=(C,q)=>{const[L,V,tt,wt]=q,Dt=Ts(x(C,tt.shape),tt,o);let ue,le;if(!n&&!s?(ue=O(Dt,V,!1,!0),le=O(L,Dt,!0,!1)):!n&&s?(ue=O(Dt,V,!1,!1),le=O(Dt,L,!0,!1)):n&&!s?(ue=O(V,Dt,!1,!0),le=O(L,Dt,!1,!1)):(ue=O(V,Dt,!0,!0),le=O(Dt,L,!0,!0)),r!=null){const Wo=_s(wt,Dt);return[ue,le,Wo]}else return[ue,le]},F={a:E,b:T,bias:_,preluActivationWeights:A},P={transposeA:n,transposeB:s,activation:o,leakyreluAlpha:i};return r==null?Ft((q,L,V)=>{const tt=b.runKernel(Bs,F,P);return V([q,L,tt]),{value:x(tt,D),gradFunc:N}})(E,T):Ft((q,L,V,tt)=>{const wt=b.runKernel(Bs,F,P);return tt([q,L,wt,V]),{value:x(wt,D),gradFunc:N}})(E,T,_)}const qm=m({fusedMatMul_:Wm});/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Km=Object.freeze(Object.defineProperty({__proto__:null,conv2d:Bm,depthwiseConv2d:Um,matMul:qm},Symbol.toStringTag,{value:"Module"}));/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function zm(e){return Ss(e,.54,.46)}const Gm=m({hammingWindow_:zm});/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Hm(e){return Ss(e,.5,.5)}const Ao=m({hannWindow_:Hm});/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Vm(e,t,n,s=!1,r=0){let o=0;const a=[];for(;o+t<=e.size;)a.push(j(e,o,t)),o+=n;if(s)for(;o<e.size;){const i=o+t-e.size,c=ft([j(e,o,t-i),Ye([i],r)]);a.push(c),o+=n}return a.length===0?Fe([],[0,t]):x(ft(a),[a.length,t])}const Mo=m({frame_:Vm});/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function jm(e,t,n,s,r=Ao){s==null&&(s=No(t));const o=Mo(e,t,n),a=S(o,r(t));return xs(a,s)}const Xm=m({stft_:jm});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Ym(e,t,n,s,r="bilinear",o=0){const a=f(e,"image","cropAndResize"),i=f(t,"boxes","cropAndResize","float32"),c=f(n,"boxInd","cropAndResize","int32"),u=i.shape[0];p(a.rank===4,()=>`Error in cropAndResize: image must be rank 4,but got rank ${a.rank}.`),p(i.rank===2&&i.shape[1]===4,()=>`Error in cropAndResize: boxes must be have size [${u},4] but had shape ${i.shape}.`),p(c.rank===1&&c.shape[0]===u,()=>`Error in cropAndResize: boxInd must be have size [${u}] but had shape ${i.shape}.`),p(s.length===2,()=>`Error in cropAndResize: cropSize must be of length 2, but got length ${s.length}.`),p(s[0]>=1&&s[1]>=1,()=>`cropSize must be atleast [1,1], but was ${s}`),p(r==="bilinear"||r==="nearest",()=>`method must be bilinear or nearest, but was ${r}`);const h={image:a,boxes:i,boxInd:c},l={method:r,extrapolationValue:o,cropSize:s};return b.runKernel(Ya,h,l)}const Zm=m({cropAndResize_:Ym});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Jm(e){const t=f(e,"image","flipLeftRight","float32");p(t.rank===4,()=>`Error in flipLeftRight: image must be rank 4,but got rank ${t.rank}.`);const n={image:t};return b.runKernel(pi,n,{})}const Qm=m({flipLeftRight_:Jm});/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function tb(e){const t=f(e,"image","grayscaleToRGB"),n=t.rank-1,s=t.shape[n];p(t.rank>=2,()=>`Error in grayscaleToRGB: images must be at least rank 2, but got rank ${t.rank}.`),p(s===1,()=>`Error in grayscaleToRGB: last dimension of a grayscale image should be size 1, but got size ${s}.`);const r=new Array(t.rank);return r.fill(1,0,n),r[n]=3,Me(t,r)}const eb=m({grayscaleToRGB_:tb});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function nb(e,t,n=0,s=.5){const r=f(e,"image","rotateWithOffset","float32");p(r.rank===4,()=>`Error in rotateWithOffset: image must be rank 4,but got rank ${r.rank}.`);const o={image:r},a={radians:t,fillValue:n,center:s};return b.runKernel(ru,o,a)}const sb=m({rotateWithOffset_:nb});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function _e(e,t,n,s,r,o){s==null&&(s=.5),r==null&&(r=Number.NEGATIVE_INFINITY),o==null&&(o=0);const a=e.shape[0];return n=Math.min(n,a),p(0<=s&&s<=1,()=>`iouThreshold must be in [0, 1], but was '${s}'`),p(e.rank===2,()=>`boxes must be a 2D tensor, but was of rank '${e.rank}'`),p(e.shape[1]===4,()=>`boxes must have 4 columns, but 2nd dimension was ${e.shape[1]}`),p(t.rank===1,()=>"scores must be a 1D tensor"),p(t.shape[0]===a,()=>`scores has incompatible shape with boxes. Expected ${a}, but was ${t.shape[0]}`),p(0<=o&&o<=1,()=>`softNmsSigma must be in [0, 1], but was '${o}'`),{maxOutputSize:n,iouThreshold:s,scoreThreshold:r,softNmsSigma:o}}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function rb(e,t,n,s=.5,r=Number.NEGATIVE_INFINITY){const o=f(e,"boxes","nonMaxSuppression","float32"),a=f(t,"scores","nonMaxSuppression","float32"),i=_e(o,a,n,s,r);n=i.maxOutputSize,s=i.iouThreshold,r=i.scoreThreshold;const c={maxOutputSize:n,iouThreshold:s,scoreThreshold:r};return b.runKernel(Zi,{boxes:o,scores:a},c)}const ob=m({nonMaxSuppression_:rb});/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function ab(e,t,n){const s=ib(e,t,n),r=s<0?-(s+1):s;e.splice(r,0,t)}function ib(e,t,n){return ub(e,t,n||cb)}function cb(e,t){return e>t?1:e<t?-1:0}function ub(e,t,n){let s=0,r=e.length,o=0,a=!1;for(;s<r;){o=s+(r-s>>>1);const i=n(t,e[o]);i>0?s=o+1:(r=o,a=!i)}return a?s:-s-1}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function lb(e,t,n,s,r){return Ns(e,t,n,s,r,0)}function hb(e,t,n,s,r,o){return Ns(e,t,n,s,r,0,!1,o,!0)}function fb(e,t,n,s,r,o){return Ns(e,t,n,s,r,o,!0)}function Ns(e,t,n,s,r,o,a=!1,i=!1,c=!1){const u=[];for(let $=0;$<t.length;$++)t[$]>r&&u.push({score:t[$],boxIndex:$,suppressBeginIndex:0});u.sort(Zs);const h=o>0?-.5/o:0,l=[],d=[];for(;l.length<n&&u.length>0;){const $=u.pop(),{score:k,boxIndex:I,suppressBeginIndex:D}=$;if(k<r)break;let E=!1;for(let T=l.length-1;T>=D;--T){const _=db(e,I,l[T]);if(_>=s){E=!0;break}if($.score=$.score*pb(s,h,_),$.score<=r)break}$.suppressBeginIndex=l.length,E||($.score===k?(l.push(I),d.push($.score)):$.score>r&&ab(u,$,Zs))}const g=l.length,w=n-g;i&&w>0&&(l.push(...new Array(w).fill(0)),d.push(...new Array(w).fill(0)));const y={selectedIndices:l};return a&&(y.selectedScores=d),c&&(y.validOutputs=g),y}function db(e,t,n){const s=e.subarray(t*4,t*4+4),r=e.subarray(n*4,n*4+4),o=Math.min(s[0],s[2]),a=Math.min(s[1],s[3]),i=Math.max(s[0],s[2]),c=Math.max(s[1],s[3]),u=Math.min(r[0],r[2]),h=Math.min(r[1],r[3]),l=Math.max(r[0],r[2]),d=Math.max(r[1],r[3]),g=(i-o)*(c-a),w=(l-u)*(d-h);if(g<=0||w<=0)return 0;const y=Math.max(o,u),$=Math.max(a,h),k=Math.min(i,l),I=Math.min(c,d),D=Math.max(k-y,0)*Math.max(I-$,0);return D/(g+w-D)}function pb(e,t,n){const s=Math.exp(t*n*n);return n<=e?s:0}function Zs(e,t){return e.score-t.score||e.score===t.score&&t.boxIndex-e.boxIndex}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */async function gb(e,t,n,s=.5,r=Number.NEGATIVE_INFINITY){const o=f(e,"boxes","nonMaxSuppressionAsync"),a=f(t,"scores","nonMaxSuppressionAsync"),i=_e(o,a,n,s,r);n=i.maxOutputSize,s=i.iouThreshold,r=i.scoreThreshold;const c=await Promise.all([o.data(),a.data()]),u=c[0],h=c[1],{selectedIndices:l}=lb(u,h,n,s,r);return o!==e&&o.dispose(),a!==t&&a.dispose(),vt(l,"int32")}const mb=gb;/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function bb(e,t,n,s=.5,r=Number.NEGATIVE_INFINITY,o=0){const a=f(e,"boxes","nonMaxSuppression"),i=f(t,"scores","nonMaxSuppression"),c=_e(a,i,n,s,r,o);n=c.maxOutputSize,s=c.iouThreshold,r=c.scoreThreshold,o=c.softNmsSigma;const u={boxes:a,scores:i},h={maxOutputSize:n,iouThreshold:s,scoreThreshold:r,softNmsSigma:o},l=b.runKernel(Qi,u,h);return{selectedIndices:l[0],selectedScores:l[1]}}const wb=m({nonMaxSuppressionWithScore_:bb});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */async function yb(e,t,n,s=.5,r=Number.NEGATIVE_INFINITY,o=0){const a=f(e,"boxes","nonMaxSuppressionAsync"),i=f(t,"scores","nonMaxSuppressionAsync"),c=_e(a,i,n,s,r,o);n=c.maxOutputSize,s=c.iouThreshold,r=c.scoreThreshold,o=c.softNmsSigma;const u=await Promise.all([a.data(),i.data()]),h=u[0],l=u[1],{selectedIndices:d,selectedScores:g}=fb(h,l,n,s,r,o);return a!==e&&a.dispose(),i!==t&&i.dispose(),{selectedIndices:vt(d,"int32"),selectedScores:vt(g)}}const $b=yb;/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function xb(e,t,n,s=.5,r=Number.NEGATIVE_INFINITY,o=!1){const a=f(e,"boxes","nonMaxSuppression"),i=f(t,"scores","nonMaxSuppression"),c=_e(a,i,n,s,r,null),u=c.maxOutputSize,h=c.iouThreshold,l=c.scoreThreshold,d={boxes:a,scores:i},g={maxOutputSize:u,iouThreshold:h,scoreThreshold:l,padToMaxOutputSize:o},w=b.runKernel(Ji,d,g);return{selectedIndices:w[0],validOutputs:w[1]}}const kb=m({nonMaxSuppressionPadded_:xb});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */async function vb(e,t,n,s=.5,r=Number.NEGATIVE_INFINITY,o=!1){const a=f(e,"boxes","nonMaxSuppressionAsync"),i=f(t,"scores","nonMaxSuppressionAsync"),c=_e(a,i,n,s,r,null),u=c.maxOutputSize,h=c.iouThreshold,l=c.scoreThreshold,[d,g]=await Promise.all([a.data(),i.data()]),{selectedIndices:w,validOutputs:y}=hb(d,g,u,h,l,o);return a!==e&&a.dispose(),i!==t&&i.dispose(),{selectedIndices:vt(w,"int32"),validOutputs:U(y,"int32")}}const Eb=vb;/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Sb(e,t,n=!1,s=!1){const r=f(e,"images","resizeBilinear");p(r.rank===3||r.rank===4,()=>`Error in resizeBilinear: x must be rank 3 or 4, but got rank ${r.rank}.`),p(t.length===2,()=>`Error in resizeBilinear: new shape must 2D, but got shape ${t}.`),p(s===!1||n===!1,()=>"Error in resizeBilinear: If halfPixelCenters is true, alignCorners must be false.");let o=r,a=!1;r.rank===3&&(a=!0,o=x(r,[1,r.shape[0],r.shape[1],r.shape[2]]));const i={images:o},c={alignCorners:n,halfPixelCenters:s,size:t},u=b.runKernel(mc,i,c);return a?x(u,[u.shape[1],u.shape[2],u.shape[3]]):u}const Tb=m({resizeBilinear_:Sb});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function _b(e,t,n=!1,s=!1){const r=f(e,"images","resizeNearestNeighbor");p(r.rank===3||r.rank===4,()=>`Error in resizeNearestNeighbor: x must be rank 3 or 4, but got rank ${r.rank}.`),p(t.length===2,()=>`Error in resizeNearestNeighbor: new shape must 2D, but got shape ${t}.`),p(r.dtype==="float32"||r.dtype==="int32",()=>"`images` must have `int32` or `float32` as dtype"),p(s===!1||n===!1,()=>"Error in resizeNearestNeighbor: If halfPixelCenters is true, alignCorners must be false.");let o=r,a=!1;r.rank===3&&(a=!0,o=x(r,[1,r.shape[0],r.shape[1],r.shape[2]]));const i={images:o},c={alignCorners:n,halfPixelCenters:s,size:t},u=b.runKernel(gc,i,c);return a?x(u,[u.shape[1],u.shape[2],u.shape[3]]):u}const Ib=m({resizeNearestNeighbor_:_b});/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Db(e,t="binary",n=!1,s=.5){const r=f(e,"image","threshold"),o=.2989,a=.587,i=.114,c=r.shape[0]*r.shape[1];let u=S(vt([s]),255),h,l,d,g;if(p(r.rank===3,()=>`Error in threshold: image must be rank 3,but got rank ${r.rank}.`),p(r.shape[2]===3||r.shape[2]===1,()=>`Error in threshold: image color channel must be equal to 3 or 1but got ${r.shape[2]}.`),p(r.dtype==="int32"||r.dtype==="float32",()=>`Error in dtype: image dtype must be int32 or float32,but got dtype ${r.dtype}.`),p(t==="otsu"||t==="binary",()=>`Method must be binary or otsu, but was ${t}`),r.shape[2]===3){[h,l,d]=Ke(r,[1,1,1],-1);const $=S(h,o),k=S(l,a),I=S(d,i);g=M(M($,k),I)}else g=e;if(t==="otsu"){const $=Jr(st(Eo(g),"int32"),ge([]),256);u=Nb($,c)}const w=n?us(g,u):vn(g,u);return st(S(w,255),"int32")}function Nb(e,t){let n=vt([-1]),s=vt([0]),r=vt([0]),o,a,i,c,u,h;for(let l=0;l<e.size-1;l++){o=j(e,0,l+1),a=j(e,l+1),u=z(K(o),t),h=z(K(a),t);const d=K(S(o,We(0,o.size)));i=z(d,K(o));const g=Ye(a.shape,o.size),w=M(We(0,a.size),g),y=S(a,w);c=z(K(y),K(a));const $=R(i,c),k=R(i,c),I=S(u,h);r=S(S(I,$),k);const D=vn(r,s);s=Ot(D,r,s),n=Ot(D,vt([l]),n)}return n}const Ab=m({threshold_:Db});/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Mb(e,t,n="nearest",s="constant",r=0,o){const a=f(e,"image","transform","float32"),i=f(t,"transforms","transform","float32");p(a.rank===4,()=>`Error in transform: image must be rank 4,but got rank ${a.rank}.`),p(i.rank===2&&(i.shape[0]===a.shape[0]||i.shape[0]===1)&&i.shape[1]===8,()=>"Error in transform: Input transform should be batch x 8 or 1 x 8"),p(o==null||o.length===2,()=>`Error in transform: outputShape must be [height, width] or null, but got ${o}.`);const c={image:a,transforms:i},u={interpolation:n,fillMode:s,fillValue:r,outputShape:o};return b.runKernel(Jc,c,u)}const Fb=m({transform_:Mb});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Bb(e,t,n){const s=f(e,"a","bandPart");p(s.rank>=2,()=>`bandPart(): Rank must be at least 2, got ${s.rank}.`);const r=s.shape,[o,a]=s.shape.slice(-2);let i,c;typeof t=="number"?(p(t%1===0,()=>`bandPart(): numLower must be an integer, got ${t}.`),p(t<=o,()=>`bandPart(): numLower (${t}) must not be greater than the number of rows (${o}).`),i=f(t<0?o:t,"numLower","bandPart")):(p(t.dtype==="int32",()=>"bandPart(): numLower's dtype must be an int32."),i=Ot(Yn(t,0),o,fn(t,o))),typeof n=="number"?(p(n%1===0,()=>`bandPart(): numUpper must be an integer, got ${n}.`),p(n<=a,()=>`bandPart(): numUpper (${n}) must not be greater than the number of columns (${a}).`),c=f(n<0?a:n,"numUpper","bandPart")):(p(n.dtype==="int32",()=>"bandPart(): numUpper's dtype must be an int32."),c=Ot(Yn(n,0),a,fn(n,a)));const u=x(We(0,o,1,"int32"),[-1,1]),h=We(0,a,1,"int32"),l=R(u,h),d=ln(us(l,i),co(l,_t(c))),g=Te([o,a],s.dtype);return x(ze(Es(x(s,[-1,o,a])).map(w=>Ot(d,w,g))),r)}const Cb=m({bandPart_:Bb});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Rb(e){let t;if(Array.isArray(e)){t=!1,p(e!=null&&e.length>0,()=>"Gram-Schmidt process: input must not be null, undefined, or empty");const r=e[0].shape[0];for(let o=1;o<e.length;++o)p(e[o].shape[0]===r,()=>`Gram-Schmidt: Non-unique lengths found in the input vectors: (${e[o].shape[0]} vs. ${r})`)}else t=!0,e=Ke(e,e.shape[0],0).map(r=>ks(r,[0]));p(e.length<=e[0].shape[0],()=>`Gram-Schmidt: Number of vectors (${e.length}) exceeds number of dimensions (${e[0].shape[0]}).`);const n=[],s=e;for(let r=0;r<e.length;++r)n.push(b.tidy(()=>{let o=s[r];if(r>0)for(let a=0;a<r;++a){const i=S(K(S(n[a],o)),n[a]);o=R(o,i)}return z(o,kn(o,"euclidean"))}));return t?ze(n,0):n}const Pb=m({gramSchmidt_:Rb});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Ob(e,t=!1){if(p(e.rank>=2,()=>`qr() requires input tensor to have a rank >= 2, but got rank ${e.rank}`),e.rank===2)return Js(e,t);{const n=e.shape.slice(0,e.shape.length-2).reduce((c,u)=>c*u),s=Es(x(e,[n,e.shape[e.shape.length-2],e.shape[e.shape.length-1]]),0),r=[],o=[];s.forEach(c=>{const[u,h]=Js(c,t);r.push(u),o.push(h)});const a=x(ze(r,0),e.shape),i=x(ze(o,0),e.shape);return[a,i]}}function Js(e,t=!1){return b.tidy(()=>{p(e.shape.length===2,()=>`qr2d() requires a 2D Tensor, but got a ${e.shape.length}D Tensor.`);const n=e.shape[0],s=e.shape[1];let r=oo(n),o=Jt(e);const a=Fe([[1]],[1,1]);let i=Jt(a);const c=n>=s?s:n;for(let u=0;u<c;++u){const h=o,l=i,d=r;[i,o,r]=b.tidy(()=>{const g=j(o,[u,u],[n-u,1]),w=kn(g),y=j(o,[u,u],[1,1]),$=Ot(vn(y,0),Fe([[-1]]),Fe([[1]])),k=R(y,S($,w)),I=z(g,k);I.shape[0]===1?i=Jt(a):i=ft([a,j(I,[1,0],[I.shape[0]-1,I.shape[1]])],0);const D=_t(z(O($,k),w)),E=j(o,[u,0],[n-u,s]),T=S(D,i),_=Zn(i);if(u===0)o=R(E,O(T,O(_,E)));else{const F=R(E,O(T,O(_,E)));o=ft([j(o,[0,0],[u,s]),F],0)}const A=Zn(T),N=j(r,[0,u],[n,r.shape[1]-u]);if(u===0)r=R(N,O(O(N,i),A));else{const F=R(N,O(O(N,i),A));r=ft([j(r,[0,0],[n,u]),F],1)}return[i,o,r]}),ht([h,l,d])}return!t&&n>s&&(r=j(r,[0,0],[n,s]),o=j(o,[0,0],[s,s])),[r,o]})}const Lb=m({qr_:Ob});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */var it;(function(e){e[e.NONE=0]="NONE",e[e.MEAN=1]="MEAN",e[e.SUM=2]="SUM",e[e.SUM_BY_NONZERO_WEIGHTS=3]="SUM_BY_NONZERO_WEIGHTS"})(it||(it={}));function Ub(e,t,n=it.SUM_BY_NONZERO_WEIGHTS){const s=f(e,"losses","computeWeightedLoss");let r=null;t!=null&&(r=f(t,"weights","computeWeightedLoss"));const o=r==null?s:S(s,r);if(n===it.NONE)return o;if(n===it.SUM)return K(o);if(n===it.MEAN){if(r==null)return hn(o);{const a=s.size/r.size,i=z(K(o),K(r));return a>1?z(i,U(a)):i}}if(n===it.SUM_BY_NONZERO_WEIGHTS){if(r==null)return z(K(o),U(s.size));{const a=S(r,Zt(s.shape)),i=st(K(wo(a,U(0))),"float32");return z(K(o),i)}}throw Error(`Unknown reduction: ${n}`)}const Bt=m({computeWeightedLoss_:Ub});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Wb(e,t,n,s=it.SUM_BY_NONZERO_WEIGHTS){const r=f(e,"labels","absoluteDifference"),o=f(t,"predictions","absoluteDifference");let a=null;n!=null&&(a=f(n,"weights","absoluteDifference")),at(r.shape,o.shape,"Error in absoluteDifference: ");const i=gt(R(r,o));return Bt(i,a,s)}const qb=m({absoluteDifference_:Wb});function Kb(e,t,n,s,r=it.SUM_BY_NONZERO_WEIGHTS){const o=f(e,"labels","cosineDistance"),a=f(t,"predictions","cosineDistance");let i=null;s!=null&&(i=f(s,"weights","cosineDistance")),at(o.shape,a.shape,"Error in cosineDistance: ");const c=U(1),u=R(c,K(S(o,a),n,!0));return Bt(u,i,r)}const zb=m({cosineDistance_:Kb});function Gb(e,t,n,s=it.SUM_BY_NONZERO_WEIGHTS){let r=f(e,"labels","hingeLoss");const o=f(t,"predictions","hingeLoss");let a=null;n!=null&&(a=f(n,"weights","hingeLoss")),at(r.shape,o.shape,"Error in hingeLoss: ");const i=U(1);r=R(S(U(2),r),i);const c=Tn(R(i,S(r,o)));return Bt(c,a,s)}const Hb=m({hingeLoss_:Gb});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Vb(e,t,n,s=1,r=it.SUM_BY_NONZERO_WEIGHTS){const o=f(e,"labels","huberLoss"),a=f(t,"predictions","huberLoss");let i=null;n!=null&&(i=f(n,"weights","huberLoss")),at(o.shape,a.shape,"Error in huberLoss: ");const c=U(s),u=gt(R(a,o)),h=fn(u,c),l=R(u,h),d=M(S(U(.5),Et(h)),S(c,l));return Bt(d,i,r)}const jb=m({huberLoss_:Vb});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Xb(e,t,n,s=1e-7,r=it.SUM_BY_NONZERO_WEIGHTS){const o=f(e,"labels","logLoss"),a=f(t,"predictions","logLoss");let i=null;n!=null&&(i=f(n,"weights","logLoss")),at(o.shape,a.shape,"Error in logLoss: ");const c=U(1),u=U(s),h=_t(S(o,Ue(M(a,u)))),l=S(R(c,o),Ue(M(R(c,a),u))),d=R(h,l);return Bt(d,i,r)}const Yb=m({logLoss_:Xb});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Zb(e,t,n,s=it.SUM_BY_NONZERO_WEIGHTS){const r=f(e,"labels","meanSquaredError"),o=f(t,"predictions","meanSquaredError");let a=null;n!=null&&(a=f(n,"weights","meanSquaredError")),at(r.shape,o.shape,"Error in meanSquaredError: ");const i=To(r,o);return Bt(i,a,s)}const Jb=m({meanSquaredError_:Zb});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Qb(e,t){const n=f(e,"labels","sigmoidCrossEntropyWithLogits"),s=f(t,"logits","sigmoidCrossEntropyWithLogits");at(n.shape,s.shape,"Error in sigmoidCrossEntropyWithLogits: ");const r=Tn(s),o=S(s,n),a=lo(ee(_t(gt(s))));return M(R(r,o),a)}function tw(e,t,n,s=0,r=it.SUM_BY_NONZERO_WEIGHTS){let o=f(e,"multiClassLabels","sigmoidCrossEntropy");const a=f(t,"logits","sigmoidCrossEntropy");let i=null;if(n!=null&&(i=f(n,"weights","sigmoidCrossEntropy")),at(o.shape,a.shape,"Error in sigmoidCrossEntropy: "),s>0){const u=U(s),h=U(1),l=U(.5);o=M(S(o,R(h,u)),S(l,u))}const c=Qb(o,a);return Bt(c,i,r)}const ew=m({sigmoidCrossEntropy_:tw});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function nw(e,t,n=-1){if(n===-1&&(n=t.rank-1),n!==t.rank-1)throw Error(`Softmax cross entropy along a non-last dimension is not yet supported. Labels / logits was rank ${t.rank} and dim was ${n}`);return Ft((r,o,a)=>{const c=fo(o,[n],!0),u=R(st(o,"float32"),c);a([r,u]);const h=_t(S(u,r));return{value:K(h,[n]),gradFunc:(g,w)=>{const[y,$]=w,k=xn(g.shape,[n]);return[S(x(g,k),R(st(y,"float32"),ee($))),S(x(g,k),R(ee($),st(y,"float32")))]}}})(e,t)}function sw(e,t,n,s=0,r=it.SUM_BY_NONZERO_WEIGHTS){let o=f(e,"onehotLabels","softmaxCrossEntropy");const a=f(t,"logits","softmaxCrossEntropy");let i=null;if(n!=null&&(i=f(n,"weights","softmaxCrossEntropy")),at(o.shape,a.shape,"Error in softmaxCrossEntropy: "),s>0){const u=U(s),h=U(1),l=U(o.shape[1]);o=M(S(o,R(h,u)),z(u,l))}const c=nw(o,a);return Bt(c,i,r)}const rw=m({softmaxCrossEntropy_:sw});/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function ow(e,t,n,s){const r=f(e,"indices","sparseFillEmptyRows","int32"),o=f(t,"values","sparseFillEmptyRows"),a=f(n,"denseShape","sparseFillEmptyRows","int32"),i=f(s,"defaultValue","sparseFillEmptyRows",o.dtype);if(r.rank!==2)throw new Error(`Indices should be Tensor2D but received shape
        ${r.shape}`);if(o.rank!==1)throw new Error(`Values should be Tensor1D but received shape ${o.shape}`);if(a.rank!==1)throw new Error(`Dense shape should be Tensor1D but received shape ${a.shape}`);if(i.rank!==0)throw new Error(`Default value should be a scalar but received shape ${i.shape}`);const c={indices:r,values:o,denseShape:a,defaultValue:i},u=b.runKernel(Pc,c);return{outputIndices:u[0],outputValues:u[1],emptyRowIndicator:u[2],reverseIndexMap:u[3]}}const aw=m({sparseFillEmptyRows_:ow});/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function iw(e,t,n){const s=f(e,"inputIndices","sparseReshape","int32"),r=f(t,"inputShape","sparseReshape","int32"),o=f(n,"newShape","sparseReshape","int32");if(s.rank!==2)throw new Error(`Input indices should be Tensor2D but received shape
        ${s.shape}`);if(r.rank!==1)throw new Error(`Input shape should be Tensor1D but received shape ${r.shape}`);if(o.rank!==1)throw new Error(`New shape should be Tensor1D but received shape ${o.shape}`);const a={inputIndices:s,inputShape:r,newShape:o},i=b.runKernel(Oc,a);return{outputIndices:i[0],outputShape:i[1]}}const cw=m({sparseReshape_:iw});/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function uw(e,t,n){const s=f(e,"data","sparseSegmentMean"),r=f(t,"indices","sparseSegmentMean","int32"),o=f(n,"segmentIds","sparseSegmentMean","int32");if(s.rank<1)throw new Error("Data should be at least 1 dimensional but received scalar");if(r.rank!==1)throw new Error(`Indices should be Tensor1D but received shape
          ${r.shape}`);if(o.rank!==1)throw new Error(`Segment ids should be Tensor1D but received shape
          ${o.shape}`);const a={data:s,indices:r,segmentIds:o};return b.runKernel(Lc,a)}const lw=m({sparseSegmentMean_:uw});/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function hw(e,t,n){const s=f(e,"data","sparseSegmentSum"),r=f(t,"indices","sparseSegmentSum","int32"),o=f(n,"segmentIds","sparseSegmentSum","int32");if(s.rank<1)throw new Error("Data should be at least 1 dimensional but received scalar");if(r.rank!==1)throw new Error(`Indices should be Tensor1D but received shape
         ${r.shape}`);if(o.rank!==1)throw new Error(`Segment ids should be Tensor1D but received shape
         ${o.shape}`);const a={data:s,indices:r,segmentIds:o};return b.runKernel(Uc,a)}const fw=m({sparseSegmentSum_:hw});/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function dw(e,t,n,s,r,o,a,i){const c=f(e,"data","stringNGrams","string");if(c.dtype!=="string")throw new Error("Data must be of datatype string");if(c.shape.length!==1)throw new Error(`Data must be a vector, saw: ${c.shape}`);const u=f(t,"dataSplits","stringNGrams");if(u.dtype!=="int32")throw new Error("Data splits must be of datatype int32");const h={separator:n,nGramWidths:s,leftPad:r,rightPad:o,padWidth:a,preserveShortSequences:i},l={data:c,dataSplits:u},d=b.runKernel(Gc,l,h);return{nGrams:d[0],nGramsSplits:d[1]}}const pw=m({stringNGrams_:dw});/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function gw(e,t,n=!0){const s=f(e,"input","stringSplit","string"),r=f(t,"delimiter","stringSplit","string");if(s.rank!==1)throw new Error(`Input should be Tensor1D but received shape ${s.shape}`);if(r.rank!==0)throw new Error(`Delimiter should be a scalar but received shape ${r.shape}`);const o={skipEmpty:n},a={input:s,delimiter:r},i=b.runKernel(Hc,a,o);return{indices:i[0],values:i[1],shape:i[2]}}const mw=m({stringSplit_:gw});/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function bw(e,t){const n=f(e,"input","stringToHashBucketFast","string"),s={numBuckets:t};if(t<=0)throw new Error("Number of buckets must be at least 1");const r={input:n};return b.runKernel(Vc,r,s)}const ww=m({stringToHashBucketFast_:bw});/**
 * @license
 * Copyright 2023 Google LLC.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function yw(e,t,n,s=!0){const r=f(e,"input","staticRegexReplace","string"),o={pattern:t,rewrite:n,replaceGlobal:s};return b.runKernel(Kc,{x:r},o)}const $w=m({staticRegexReplace_:yw});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const xw={fft:$s,ifft:dn,rfft:xs,irfft:So},kw={hammingWindow:Gm,hannWindow:Ao,frame:Mo,stft:Xm},vw={flipLeftRight:Qm,grayscaleToRGB:eb,resizeNearestNeighbor:Ib,resizeBilinear:Tb,rotateWithOffset:sb,cropAndResize:Zm,nonMaxSuppression:ob,nonMaxSuppressionAsync:mb,nonMaxSuppressionWithScore:wb,nonMaxSuppressionWithScoreAsync:$b,nonMaxSuppressionPadded:kb,nonMaxSuppressionPaddedAsync:Eb,threshold:Ab,transform:Fb},Ew={bandPart:Cb,gramSchmidt:Pb,qr:Lb},Sw={absoluteDifference:qb,computeWeightedLoss:Bt,cosineDistance:zb,hingeLoss:Hb,huberLoss:jb,logLoss:Yb,meanSquaredError:Jb,sigmoidCrossEntropy:ew,softmaxCrossEntropy:rw},Tw={sparseFillEmptyRows:aw,sparseReshape:cw,sparseSegmentMean:lw,sparseSegmentSum:fw},_w={stringNGrams:pw,stringSplit:mw,stringToHashBucketFast:ww,staticRegexReplace:$w};/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class Fo{getClassName(){return this.constructor.className}static fromConfig(t,n){return new t(n)}}class Ct{constructor(){this.classNameMap={}}static getMap(){return Ct.instance==null&&(Ct.instance=new Ct),Ct.instance}static register(t){Ct.getMap().classNameMap[t.className]=[t,t.fromConfig]}}function Bo(e){p(e.className!=null,()=>"Class being registered does not have the static className property defined."),p(typeof e.className=="string",()=>"className is required to be a string, but got type "+typeof e.className),p(e.className.length>0,()=>"Class being registered has an empty-string as its className, which is disallowed."),Ct.register(e)}const a0=Object.freeze(Object.defineProperty({__proto__:null,Serializable:Fo,SerializationMap:Ct,registerClass:Bo},Symbol.toStringTag,{value:"Module"}));/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class ce extends Fo{minimize(t,n=!1,s){const{value:r,grads:o}=this.computeGradients(t,s);if(s!=null){const a=s.map(i=>({name:i.name,tensor:o[i.name]}));this.applyGradients(a)}else this.applyGradients(o);return ht(o),n?r:(r.dispose(),null)}get iterations(){return this.iterations_==null&&(this.iterations_=0),this.iterations_}incrementIterations(){this.iterations_=this.iterations+1}computeGradients(t,n){return hd(t,n)}dispose(){this.iterations_!=null&&ht(this.iterations_)}async saveIterations(){return this.iterations_==null&&(this.iterations_=0),{name:"iter",tensor:U(this.iterations_,"int32")}}async getWeights(){throw new Error("getWeights() is not implemented for this optimizer yet.")}async setWeights(t){throw new Error(`setWeights() is not implemented for this optimizer class ${this.getClassName()}`)}async extractIterations(t){return this.iterations_=(await t[0].tensor.data())[0],t.slice(1)}}Object.defineProperty(ce,Symbol.hasInstance,{value:e=>e.minimize!=null&&e.computeGradients!=null&&e.applyGradients!=null});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class Iw extends ce{static get className(){return"Adadelta"}constructor(t,n,s=null){super(),this.learningRate=t,this.rho=n,this.epsilon=s,this.accumulatedGrads=[],this.accumulatedUpdates=[],s==null&&(this.epsilon=b.backend.epsilon())}applyGradients(t){(Array.isArray(t)?t.map(s=>s.name):Object.keys(t)).forEach((s,r)=>{const o=b.registeredVariables[s],a=!1;this.accumulatedGrads[r]==null&&(this.accumulatedGrads[r]={originalName:`${s}/accum_grad`,variable:J(()=>mt(o).variable(a))}),this.accumulatedUpdates[r]==null&&(this.accumulatedUpdates[r]={originalName:`${s}/accum_var`,variable:J(()=>mt(o).variable(a))});const i=Array.isArray(t)?t[r].tensor:t[s];if(i==null)return;const c=this.accumulatedGrads[r].variable,u=this.accumulatedUpdates[r].variable;J(()=>{const h=M(S(c,this.rho),S(Et(i),1-this.rho)),l=S(z(Mt(M(u,this.epsilon)),Mt(M(c,this.epsilon))),i),d=M(S(u,this.rho),S(Et(l),1-this.rho));c.assign(h),u.assign(d);const g=M(S(l,-this.learningRate),o);o.assign(g)})}),this.incrementIterations()}dispose(){this.accumulatedUpdates!=null&&(ht(this.accumulatedGrads.map(t=>t.variable)),ht(this.accumulatedUpdates.map(t=>t.variable)))}async getWeights(){const t=[...this.accumulatedGrads,...this.accumulatedUpdates];return[await this.saveIterations()].concat(t.map(n=>({name:n.originalName,tensor:n.variable})))}async setWeights(t){t=await this.extractIterations(t);const n=t.length/2,s=!1;this.accumulatedGrads=t.slice(0,n).map(r=>({originalName:r.name,variable:r.tensor.variable(s)})),this.accumulatedUpdates=t.slice(n,n*2).map(r=>({originalName:r.name,variable:r.tensor.variable(s)}))}getConfig(){return{learningRate:this.learningRate,rho:this.rho,epsilon:this.epsilon}}static fromConfig(t,n){return new t(n.learningRate,n.rho,n.epsilon)}}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class Dw extends ce{static get className(){return"Adagrad"}constructor(t,n=.1){super(),this.learningRate=t,this.initialAccumulatorValue=n,this.accumulatedGrads=[]}applyGradients(t){(Array.isArray(t)?t.map(s=>s.name):Object.keys(t)).forEach((s,r)=>{const o=b.registeredVariables[s];this.accumulatedGrads[r]==null&&(this.accumulatedGrads[r]={originalName:`${s}/accumulator`,variable:J(()=>Ye(o.shape,this.initialAccumulatorValue).variable(!1))});const a=Array.isArray(t)?t[r].tensor:t[s];if(a==null)return;const i=this.accumulatedGrads[r].variable;J(()=>{const c=M(i,Et(a));i.assign(c);const u=M(S(z(a,Mt(M(c,b.backend.epsilon()))),-this.learningRate),o);o.assign(u)})}),this.incrementIterations()}dispose(){this.accumulatedGrads!=null&&ht(this.accumulatedGrads.map(t=>t.variable))}async getWeights(){return[await this.saveIterations()].concat(this.accumulatedGrads.map(t=>({name:t.originalName,tensor:t.variable})))}async setWeights(t){t=await this.extractIterations(t);const n=!1;this.accumulatedGrads=t.map(s=>({originalName:s.name,variable:s.tensor.variable(n)}))}getConfig(){return{learningRate:this.learningRate,initialAccumulatorValue:this.initialAccumulatorValue}}static fromConfig(t,n){return new t(n.learningRate,n.initialAccumulatorValue)}}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class Nw extends ce{static get className(){return"Adam"}constructor(t,n,s,r=null){super(),this.learningRate=t,this.beta1=n,this.beta2=s,this.epsilon=r,this.accumulatedFirstMoment=[],this.accumulatedSecondMoment=[],J(()=>{this.accBeta1=U(n).variable(),this.accBeta2=U(s).variable()}),r==null&&(this.epsilon=b.backend.epsilon())}applyGradients(t){const n=Array.isArray(t)?t.map(s=>s.name):Object.keys(t);J(()=>{const s=R(1,this.accBeta1),r=R(1,this.accBeta2);n.forEach((o,a)=>{const i=b.registeredVariables[o],c=!1;this.accumulatedFirstMoment[a]==null&&(this.accumulatedFirstMoment[a]={originalName:`${o}/m`,variable:J(()=>mt(i).variable(c))}),this.accumulatedSecondMoment[a]==null&&(this.accumulatedSecondMoment[a]={originalName:`${o}/v`,variable:J(()=>mt(i).variable(c))});const u=Array.isArray(t)?t[a].tensor:t[o];if(u==null)return;const h=this.accumulatedFirstMoment[a].variable,l=this.accumulatedSecondMoment[a].variable,d=M(S(h,this.beta1),S(u,1-this.beta1)),g=M(S(l,this.beta2),S(Et(u),1-this.beta2)),w=z(d,s),y=z(g,r);h.assign(d),l.assign(g);const $=M(S(z(w,M(Mt(y),this.epsilon)),-this.learningRate),i);i.assign($)}),this.accBeta1.assign(S(this.accBeta1,this.beta1)),this.accBeta2.assign(S(this.accBeta2,this.beta2))}),this.incrementIterations()}dispose(){this.accBeta1.dispose(),this.accBeta2.dispose(),this.accumulatedFirstMoment!=null&&ht(this.accumulatedFirstMoment.map(t=>t.variable)),this.accumulatedSecondMoment!=null&&ht(this.accumulatedSecondMoment.map(t=>t.variable))}async getWeights(){const t=[...this.accumulatedFirstMoment,...this.accumulatedSecondMoment];return[await this.saveIterations()].concat(t.map(n=>({name:n.originalName,tensor:n.variable})))}async setWeights(t){t=await this.extractIterations(t),J(()=>{this.accBeta1.assign(Le(this.beta1,this.iterations_+1)),this.accBeta2.assign(Le(this.beta2,this.iterations_+1))});const n=t.length/2,s=!1;this.accumulatedFirstMoment=t.slice(0,n).map(r=>({originalName:r.name,variable:r.tensor.variable(s)})),this.accumulatedSecondMoment=t.slice(n,n*2).map(r=>({originalName:r.name,variable:r.tensor.variable(s)}))}getConfig(){return{learningRate:this.learningRate,beta1:this.beta1,beta2:this.beta2,epsilon:this.epsilon}}static fromConfig(t,n){return new t(n.learningRate,n.beta1,n.beta2,n.epsilon)}}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class Aw extends ce{static get className(){return"Adamax"}constructor(t,n,s,r=null,o=0){super(),this.learningRate=t,this.beta1=n,this.beta2=s,this.epsilon=r,this.decay=o,this.accumulatedFirstMoment=[],this.accumulatedWeightedInfNorm=[],J(()=>{this.iteration=U(0).variable(),this.accBeta1=U(n).variable()}),r==null&&(this.epsilon=b.backend.epsilon())}applyGradients(t){const n=Array.isArray(t)?t.map(s=>s.name):Object.keys(t);J(()=>{const s=R(1,this.accBeta1),r=z(-this.learningRate,M(S(this.iteration,this.decay),1));n.forEach((o,a)=>{const i=b.registeredVariables[o],c=!1;this.accumulatedFirstMoment[a]==null&&(this.accumulatedFirstMoment[a]={originalName:`${o}/m`,variable:mt(i).variable(c)}),this.accumulatedWeightedInfNorm[a]==null&&(this.accumulatedWeightedInfNorm[a]={originalName:`${o}/v`,variable:mt(i).variable(c)});const u=Array.isArray(t)?t[a].tensor:t[o];if(u==null)return;const h=this.accumulatedFirstMoment[a].variable,l=this.accumulatedWeightedInfNorm[a].variable,d=M(S(h,this.beta1),S(u,1-this.beta1)),g=S(l,this.beta2),w=gt(u),y=bo(g,w);h.assign(d),l.assign(y);const $=M(S(z(r,s),z(d,M(y,this.epsilon))),i);i.assign($)}),this.iteration.assign(M(this.iteration,1)),this.accBeta1.assign(S(this.accBeta1,this.beta1))}),this.incrementIterations()}dispose(){this.accBeta1.dispose(),this.iteration.dispose(),this.accumulatedFirstMoment!=null&&ht(this.accumulatedFirstMoment.map(t=>t.variable)),this.accumulatedWeightedInfNorm!=null&&ht(this.accumulatedWeightedInfNorm.map(t=>t.variable))}async getWeights(){throw new Error("getWeights() is not implemented for Adamax yet.")}async setWeights(t){throw new Error("setWeights() is not implemented for Adamax yet.")}getConfig(){return{learningRate:this.learningRate,beta1:this.beta1,beta2:this.beta2,epsilon:this.epsilon,decay:this.decay}}static fromConfig(t,n){return new t(n.learningRate,n.beta1,n.beta2,n.epsilon,n.decay)}}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class Co extends ce{static get className(){return"SGD"}constructor(t){super(),this.learningRate=t,this.setLearningRate(t)}applyGradients(t){(Array.isArray(t)?t.map(s=>s.name):Object.keys(t)).forEach((s,r)=>{const o=Array.isArray(t)?t[r].tensor:t[s];if(o==null)return;const a=b.registeredVariables[s];J(()=>{const i=M(S(this.c,o),a);a.assign(i)})}),this.incrementIterations()}setLearningRate(t){this.learningRate=t,this.c!=null&&this.c.dispose(),this.c=wl(U(-t))}dispose(){this.c.dispose()}async getWeights(){return[await this.saveIterations()]}async setWeights(t){if(t=await this.extractIterations(t),t.length!==0)throw new Error("SGD optimizer does not have settable weights.")}getConfig(){return{learningRate:this.learningRate}}static fromConfig(t,n){return new t(n.learningRate)}}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class Mw extends Co{static get className(){return"Momentum"}constructor(t,n,s=!1){super(t),this.learningRate=t,this.momentum=n,this.useNesterov=s,this.accumulations=[],this.m=U(this.momentum)}applyGradients(t){(Array.isArray(t)?t.map(s=>s.name):Object.keys(t)).forEach((s,r)=>{const o=b.registeredVariables[s];this.accumulations[r]==null&&(this.accumulations[r]={originalName:`${s}/momentum`,variable:J(()=>mt(o).variable(!1))});const a=this.accumulations[r].variable,i=Array.isArray(t)?t[r].tensor:t[s];i!=null&&J(()=>{let c;const u=M(S(this.m,a),i);this.useNesterov?c=M(S(this.c,M(i,S(u,this.m))),o):c=M(S(this.c,u),o),a.assign(u),o.assign(c)})}),this.incrementIterations()}dispose(){this.m.dispose(),this.accumulations!=null&&ht(this.accumulations.map(t=>t.variable))}setMomentum(t){this.momentum=t}async getWeights(){return[await this.saveIterations()].concat(this.accumulations.map(t=>({name:t.originalName,tensor:t.variable})))}async setWeights(t){t=await this.extractIterations(t);const n=!1;this.accumulations=t.map(s=>({originalName:s.name,variable:s.tensor.variable(n)}))}getConfig(){return{learningRate:this.learningRate,momentum:this.momentum,useNesterov:this.useNesterov}}static fromConfig(t,n){return new t(n.learningRate,n.momentum,n.useNesterov)}}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class Fw extends ce{static get className(){return"RMSProp"}constructor(t,n=.9,s=0,r=null,o=!1){if(super(),this.learningRate=t,this.decay=n,this.momentum=s,this.epsilon=r,this.accumulatedMeanSquares=[],this.accumulatedMoments=[],this.accumulatedMeanGrads=[],this.centered=o,r==null&&(this.epsilon=b.backend.epsilon()),t==null)throw new Error("learningRate for RMSPropOptimizer must be defined.")}applyGradients(t){(Array.isArray(t)?t.map(s=>s.name):Object.keys(t)).forEach((s,r)=>{const o=b.registeredVariables[s],a=!1;this.accumulatedMeanSquares[r]==null&&(this.accumulatedMeanSquares[r]={originalName:`${s}/rms`,variable:J(()=>mt(o).variable(a))}),this.accumulatedMoments[r]==null&&(this.accumulatedMoments[r]={originalName:`${s}/momentum`,variable:J(()=>mt(o).variable(a))}),this.accumulatedMeanGrads[r]==null&&this.centered&&(this.accumulatedMeanGrads[r]={originalName:`${s}/mg`,variable:J(()=>mt(o).variable(a))});const i=Array.isArray(t)?t[r].tensor:t[s];if(i==null)return;const c=this.accumulatedMeanSquares[r].variable,u=this.accumulatedMoments[r].variable;J(()=>{const h=M(S(c,this.decay),S(Et(i),1-this.decay));if(this.centered){const l=this.accumulatedMeanGrads[r].variable,d=M(S(l,this.decay),S(i,1-this.decay)),g=z(S(i,this.learningRate),Mt(R(h,M(Et(d),this.epsilon)))),w=M(S(u,this.momentum),g);c.assign(h),l.assign(d),u.assign(w);const y=R(o,w);o.assign(y)}else{const l=M(S(c,this.decay),S(Et(i),1-this.decay)),d=M(S(u,this.momentum),z(S(i,this.learningRate),Mt(M(l,this.epsilon))));c.assign(l),u.assign(d);const g=R(o,d);o.assign(g)}})}),this.incrementIterations()}dispose(){this.accumulatedMeanSquares!=null&&ht(this.accumulatedMeanSquares.map(t=>t.variable)),this.accumulatedMeanGrads!=null&&this.centered&&ht(this.accumulatedMeanGrads.map(t=>t.variable)),this.accumulatedMoments!=null&&ht(this.accumulatedMoments.map(t=>t.variable))}async getWeights(){const t=[...this.accumulatedMeanSquares,...this.accumulatedMoments];return this.centered&&t.push(...this.accumulatedMeanGrads),[await this.saveIterations()].concat(t.map(n=>({name:n.originalName,tensor:n.variable})))}async setWeights(t){t=await this.extractIterations(t);const n=this.centered?t.length/3:t.length/2,s=!1;this.accumulatedMeanSquares=t.slice(0,n).map(r=>({originalName:r.name,variable:r.tensor.variable(s)})),this.accumulatedMoments=t.slice(n,n*2).map(r=>({originalName:r.name,variable:r.tensor.variable(s)})),this.centered&&(this.accumulatedMeanGrads=t.slice(n*2,n*3).map(r=>({originalName:r.name,variable:r.tensor.variable(s)})))}getConfig(){return{learningRate:this.learningRate,decay:this.decay,momentum:this.momentum,epsilon:this.epsilon,centered:this.centered}}static fromConfig(t,n){return new t(n.learningRate,n.decay,n.momentum,n.epsilon,n.centered)}}/**
 * @license
 * Copyright 2022 Google LLC.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Bw=[Iw,Dw,Nw,Aw,Mw,Fw,Co];function Cw(){for(const e of Bw)Bo(e)}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Rw="model",Pw=".json",Ow=".weights.bin";function Qs(e){return new Promise(t=>setTimeout(t)).then(e)}class se{constructor(t){if(!B().getBool("IS_BROWSER"))throw new Error("browserDownloads() cannot proceed because the current environment is not a browser.");t.startsWith(se.URL_SCHEME)&&(t=t.slice(se.URL_SCHEME.length)),(t==null||t.length===0)&&(t=Rw),this.modelJsonFileName=t+Pw,this.weightDataFileName=t+Ow}async save(t){if(typeof document>"u")throw new Error("Browser downloads are not supported in this environment since `document` is not present");const n=It.join(t.weightData),s=window.URL.createObjectURL(new Blob([n],{type:"application/octet-stream"}));if(t.modelTopology instanceof ArrayBuffer)throw new Error("BrowserDownloads.save() does not support saving model topology in binary formats yet.");{const r=[{paths:["./"+this.weightDataFileName],weights:t.weightSpecs}],o=Pr(t,r),a=window.URL.createObjectURL(new Blob([JSON.stringify(o)],{type:"application/json"})),i=this.modelJsonAnchor==null?document.createElement("a"):this.modelJsonAnchor;if(i.download=this.modelJsonFileName,i.href=a,await Qs(()=>i.dispatchEvent(new MouseEvent("click"))),t.weightData!=null){const c=this.weightDataAnchor==null?document.createElement("a"):this.weightDataAnchor;c.download=this.weightDataFileName,c.href=s,await Qs(()=>c.dispatchEvent(new MouseEvent("click")))}return{modelArtifactsInfo:Xe(t)}}}}se.URL_SCHEME="downloads://";class Lw{constructor(t){if(t==null||t.length<1)throw new Error(`When calling browserFiles, at least 1 file is required, but received ${t}`);this.jsonFile=t[0],this.weightsFiles=t.slice(1)}async load(){return new Promise((t,n)=>{const s=new FileReader;s.onload=r=>{const o=JSON.parse(r.target.result),a=o.modelTopology;if(a==null){n(new Error(`modelTopology field is missing from file ${this.jsonFile.name}`));return}if(o.weightsManifest==null){n(new Error(`weightManifest field is missing from file ${this.jsonFile.name}`));return}if(this.weightsFiles.length===0){t({modelTopology:a});return}const c=is(o,u=>this.loadWeights(u));t(c)},s.onerror=r=>n(`Failed to read model topology and weights manifest JSON from file '${this.jsonFile.name}'. BrowserFiles supports loading Keras-style tf.Model artifacts only.`),s.readAsText(this.jsonFile)})}loadWeights(t){const n=[],s=[];for(const a of t)n.push(...a.weights),s.push(...a.paths);const r=this.checkManifestAndWeightFiles(t),o=s.map(a=>this.loadWeightsFile(a,r[a]));return Promise.all(o).then(a=>[n,a])}loadWeightsFile(t,n){return new Promise((s,r)=>{const o=new FileReader;o.onload=a=>{const i=a.target.result;s(i)},o.onerror=a=>r(`Failed to weights data from file of path '${t}'.`),o.readAsArrayBuffer(n)})}checkManifestAndWeightFiles(t){const n=[],s=this.weightsFiles.map(o=>Ys(o.name)),r={};for(const o of t)o.paths.forEach(a=>{const i=Ys(a);if(n.indexOf(i)!==-1)throw new Error(`Duplicate file basename found in weights manifest: '${i}'`);if(n.push(i),s.indexOf(i)===-1)throw new Error(`Weight file with basename '${i}' is not provided.`);r[a]=this.weightsFiles[s.indexOf(i)]});if(n.length!==this.weightsFiles.length)throw new Error(`Mismatch in the number of files in weights manifest (${n.length}) and the number of weight files provided (${this.weightsFiles.length}).`);return r}}const Uw=e=>B().getBool("IS_BROWSER")&&!Array.isArray(e)&&e.startsWith(se.URL_SCHEME)?Ww(e.slice(se.URL_SCHEME.length)):null;X.registerSaveRouter(Uw);function Ww(e="model"){return new se(e)}function qw(e){return new Lw(e)}/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function tr(e,t,n,s){a(e),n=n??0,s=s??1,i(n,s);let r=0;const o=c=>(c.then(u=>{const h=n+ ++r/e.length*(s-n);return t(h),u}),c);function a(c){p(c!=null&&Array.isArray(c)&&c.length>0,()=>"promises must be a none empty array")}function i(c,u){p(c>=0&&c<=1,()=>`Progress fraction must be in range [0, 1], but got startFraction ${c}`),p(u>=0&&u<=1,()=>`Progress fraction must be in range [0, 1], but got endFraction ${u}`),p(u>=c,()=>`startFraction must be no more than endFraction, but got startFraction ${c} and endFraction ${u}`)}return Promise.all(e.map(o))}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */async function Ro(e,t){t==null&&(t={});const n=t.fetchFunc==null?B().platform.fetch:t.fetchFunc,s=e.map(l=>n(l,t.requestInit,{isBinary:!0})),r=0,o=.5,i=(t.onProgress==null?await Promise.all(s):await tr(s,t.onProgress,r,o)).map(l=>l.arrayBuffer()),c=.5,u=1;return t.onProgress==null?await Promise.all(i):await tr(i,t.onProgress,c,u)}async function Kw(e,t="",n,s){return Po(a=>Ro(a,{requestInit:s}))(e,t,n)}function Po(e){return async(t,n="",s)=>{const r=t.map(()=>!1),o={},a=s!=null?s.map(()=>!1):[],i=[];if(t.forEach((g,w)=>{let y=0;g.weights.forEach($=>{const k="quantization"in $?$.quantization.dtype:$.dtype,I=Kn[k]*Z($.shape),D=()=>{r[w]=!0,o[w]==null&&(o[w]=[]),o[w].push({manifestEntry:$,groupOffset:y,sizeBytes:I})};s!=null?s.forEach((E,T)=>{E===$.name&&(D(),a[T]=!0)}):D(),i.push($.name),y+=I})}),!a.every(g=>g)){const g=s.filter((w,y)=>!a[y]);throw new Error(`Could not find weights in manifest with names: ${g.join(", ")}. 
Manifest JSON has weights with names: ${i.join(", ")}.`)}const c=r.reduce((g,w,y)=>(w&&g.push(y),g),[]),u=[];c.forEach(g=>{t[g].paths.forEach(w=>{const y=n+(n.endsWith("/")?"":"/")+w;u.push(y)})});const h=await e(u),l={};let d=0;return c.forEach(g=>{const w=t[g].paths.length,y=new It(h.slice(d,d+w));o[g].forEach(k=>{const I=y.slice(k.groupOffset,k.groupOffset+k.sizeBytes),D=Rr(I,[k.manifestEntry]);for(const E in D)l[E]=D[E]}),d+=w}),l}}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const zw="application/octet-stream",Gw="application/json";class As{constructor(t,n){if(this.DEFAULT_METHOD="POST",n==null&&(n={}),this.weightPathPrefix=n.weightPathPrefix,this.onProgress=n.onProgress,this.weightUrlConverter=n.weightUrlConverter,n.fetchFunc!=null?(p(typeof n.fetchFunc=="function",()=>"Must pass a function that matches the signature of `fetch` (see https://developer.mozilla.org/en-US/docs/Web/API/Fetch_API)"),this.fetch=n.fetchFunc):this.fetch=B().platform.fetch,p(t!=null&&t.length>0,()=>"URL path for http must not be null, undefined or empty."),Array.isArray(t)&&p(t.length===2,()=>`URL paths for http must have a length of 2, (actual length is ${t.length}).`),this.path=t,n.requestInit!=null&&n.requestInit.body!=null)throw new Error("requestInit is expected to have no pre-existing body, but has one.");this.requestInit=n.requestInit||{}}async save(t){if(t.modelTopology instanceof ArrayBuffer)throw new Error("BrowserHTTPRequest.save() does not support saving model topology in binary formats yet.");const n=Object.assign({method:this.DEFAULT_METHOD},this.requestInit);n.body=new FormData;const s=[{paths:["./model.weights.bin"],weights:t.weightSpecs}],r=Pr(t,s);if(n.body.append("model.json",new Blob([JSON.stringify(r)],{type:Gw}),"model.json"),t.weightData!=null){const a=It.join(t.weightData);n.body.append("model.weights.bin",new Blob([a],{type:zw}),"model.weights.bin")}const o=await this.fetch(this.path,n);if(o.ok)return{modelArtifactsInfo:Xe(t),responses:[o]};throw new Error(`BrowserHTTPRequest.save() failed due to HTTP response status ${o.status}.`)}async load(){const t=await this.fetch(this.path,this.requestInit);if(!t.ok)throw new Error(`Request to ${this.path} failed with status code ${t.status}. Please verify this URL points to the model JSON of the model to load.`);let n;try{n=await t.json()}catch{let a=`Failed to parse model JSON of response from ${this.path}.`;throw this.path.endsWith(".pb")?a+=" Your path contains a .pb file extension. Support for .pb models have been removed in TensorFlow.js 1.0 in favor of .json models. You can re-convert your Python TensorFlow model using the TensorFlow.js 1.0 conversion scripts or you can convert your.pb models with the 'pb2json'NPM script in the tensorflow/tfjs-converter repository.":a+=" Please make sure the server is serving valid JSON for this request.",new Error(a)}const s=n.modelTopology,r=n.weightsManifest;if(s==null&&r==null)throw new Error(`The JSON from HTTP path ${this.path} contains neither model topology or manifest for weights.`);return is(n,o=>this.loadWeights(o))}async loadWeights(t){const n=Array.isArray(this.path)?this.path[1]:this.path,[s,r]=Hw(n),o=this.weightPathPrefix||s,a=Lr(t),i=[],c=[];for(const h of t)for(const l of h.paths)this.weightUrlConverter!=null?c.push(this.weightUrlConverter(l)):i.push(o+l+r);this.weightUrlConverter&&i.push(...await Promise.all(c));const u=await Ro(i,{requestInit:this.requestInit,fetchFunc:this.fetch,onProgress:this.onProgress});return[a,u]}}As.URL_SCHEME_REGEX=/^https?:\/\//;function Hw(e){const t=e.lastIndexOf("/"),n=e.lastIndexOf("?"),s=e.substring(0,t),r=n>t?e.substring(n):"";return[s+"/",r]}function Jn(e){return e.match(As.URL_SCHEME_REGEX)!=null}const Oo=(e,t)=>{if(typeof fetch>"u"&&(t==null||t.fetchFunc==null))return null;{let n=!0;if(Array.isArray(e)?n=e.every(s=>Jn(s)):n=Jn(e),n)return Ms(e,t)}return null};X.registerSaveRouter(Oo);X.registerLoadRouter(Oo);function Ms(e,t){return new As(e,t)}function Vw(e,t){return Ms(e,t)}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class Mn{constructor(t){this.modelArtifacts=t}load(){return this.modelArtifacts}}class Lo{constructor(t){this.saveHandler=t}save(t){return this.saveHandler(t)}}class jw{constructor(t){t.load&&(this.load=()=>Promise.resolve(t.load())),t.save&&(this.save=n=>Promise.resolve(t.save(n)))}}function Xw(e,t,n,s){const r=arguments;return new jw(Uo(...r))}function Uo(e,t,n,s){return arguments.length===1?e.modelTopology!=null||e.weightSpecs!=null?new Mn(e):(console.warn("Please call tf.io.fromMemory() with only one argument. The argument should be of type ModelArtifacts. The multi-argument signature of tf.io.fromMemory() has been deprecated and will be removed in a future release."),new Mn({modelTopology:e})):(console.warn("Please call tf.io.fromMemory() with only one argument. The argument should be of type ModelArtifacts. The multi-argument signature of tf.io.fromMemory() has been deprecated and will be removed in a future release."),new Mn({modelTopology:e,weightSpecs:t,weightData:n,trainingConfig:s}))}function Yw(e){return new Lo(e)}function Zw(e){return new Lo(e)}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const i0=Object.freeze(Object.defineProperty({__proto__:null,CompositeArrayBuffer:It,browserFiles:qw,browserHTTPRequest:Vw,concatenateArrayBuffers:qu,copyModel:ll,decodeWeights:Rr,encodeWeights:Ou,fromMemory:Xw,fromMemorySync:Uo,getLoadHandlers:Yu,getModelArtifactsForJSON:is,getModelArtifactsForJSONSync:Or,getModelArtifactsInfoForJSON:Xe,getSaveHandlers:Xu,getWeightSpecs:Lr,http:Ms,isHTTPScheme:Jn,listModels:cl,loadWeights:Kw,moveModel:hl,registerLoadRouter:ju,registerSaveRouter:Vu,removeModel:ul,weightsLoaderFactory:Po,withSaveHandler:Yw,withSaveHandlerSync:Zw},Symbol.toStringTag,{value:"Module"}));/**
 * @license
 * Copyright 2017 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */Cw();/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const c0=Object.freeze(Object.defineProperty({__proto__:null,OP_SCOPE_SUFFIX:Cr,abs:gt,acos:Sl,acosh:_l,add:M,addN:Dl,all:Al,any:Fl,argMax:Cl,argMin:Pl,asin:Ll,asinh:Wl,atan:Kl,atan2:Gl,atanh:Vl,avgPool:Yr,avgPool3d:rh,basicLSTMCell:hh,batchNorm:yn,batchNorm2d:mh,batchNorm3d:wh,batchNorm4d:$h,batchToSpaceND:Zr,bincount:Jr,bitwiseAnd:vh,booleanMaskAsync:gm,broadcastArgs:Sh,broadcastTo:nn,buffer:At,cast:st,ceil:Ih,clipByValue:Nh,clone:Jt,complex:Ut,concat:ft,concat1d:Mh,concat2d:Bh,concat3d:Rh,concat4d:Oh,conv1d:Wh,conv2d:$n,conv2dTranspose:zh,conv3d:Hh,conv3dTranspose:Yh,cos:Jh,cosh:tf,cosineWindow:Ss,cumprod:nf,cumsum:rf,denseBincount:af,depthToSpace:uf,depthwiseConv2d:cs,diag:ff,dilation2d:pf,div:z,divNoNan:$f,dot:kf,dropout:Im,einsum:Ef,elu:no,enclosingPowerOfTwo:No,ensureShape:_f,equal:eo,erf:Df,euclideanNorm:Lf,exp:ee,expandDims:Ht,expm1:Kf,eye:oo,fft:$s,fill:Ye,floor:ao,floorDiv:jr,fused:Km,gather:io,gatherND:Sm,greater:vn,greaterEqual:co,ifft:dn,imag:En,image:vw,inTopKAsync:Nm,irfft:So,isFinite:Jf,isInf:td,isNaN:nd,leakyRelu:uo,less:Yn,lessEqual:us,linalg:Ew,linspace:ad,localResponseNormalization:cd,log:Ue,log1p:lo,logSigmoid:gd,logSoftmax:wd,logSumExp:fo,logicalAnd:ln,logicalNot:po,logicalOr:go,logicalXor:Ed,losses:Sw,lowerBound:Td,matMul:O,max:ye,maxPool:mo,maxPool3d:Dd,maxPoolWithArgmax:Ad,maximum:bo,mean:hn,meshgrid:Bd,min:Xn,minimum:fn,mirrorPad:Pd,mod:Ld,moments:Wd,movingAverage:wm,mul:S,multiRNNCell:Kd,multinomial:Gd,neg:_t,norm:kn,notEqual:wo,oneHot:jd,ones:Zt,onesLike:Yd,op:m,outerProduct:Jd,pad:Ze,pad1d:ep,pad2d:sp,pad3d:op,pad4d:ip,pool:fp,pow:Le,prelu:$o,print:Vr,prod:gp,raggedGather:bp,raggedRange:yp,raggedTensorToTensor:xp,rand:vp,randomGamma:Kp,randomNormal:ko,randomStandardNormal:Hp,randomUniform:ys,randomUniformInt:Xp,range:We,real:qe,reciprocal:Jp,relu:Tn,relu6:vo,reshape:x,reverse:ne,reverse1d:sg,reverse2d:og,reverse3d:ig,reverse4d:ug,rfft:xs,round:Eo,rsqrt:fg,scalar:U,scatterND:$m,searchSorted:ls,selu:pg,separableConv2d:mg,setdiff1dAsync:wg,sigmoid:we,sign:$g,signal:kw,sin:kg,sinh:Eg,slice:j,slice1d:Tg,slice2d:Ig,slice3d:Ng,slice4d:Mg,softmax:Bg,softplus:ho,spaceToBatchND:yo,sparse:Tw,sparseToDense:vm,spectral:xw,split:Ke,sqrt:Mt,square:Et,squaredDifference:To,squeeze:ks,stack:ze,step:_o,stridedSlice:Gg,string:_w,sub:R,sum:K,tan:Vg,tanh:jn,tensor:ge,tensor1d:vt,tensor2d:Fe,tensor3d:jg,tensor4d:Xg,tensor5d:Yg,tensor6d:Zg,tensorScatterUpdate:tm,tile:Me,topk:nm,transpose:Zn,truncatedNormal:rm,unique:am,unsortedSegmentSum:cm,unstack:Es,upperBound:lm,variable:hm,where:Ot,whereAsync:Do,zeros:Te,zerosLike:mt},Symbol.toStringTag,{value:"Module"}));export{t0 as $,Nw as A,Z1 as B,so as C,o1 as D,b as E,y1 as F,Jg as G,Tt as H,Nf as I,wn as J,Xl as K,Xr as L,Mw as M,V1 as N,Y1 as O,jl as P,j1 as Q,Fw as R,Co as S,nt as T,th as U,zt as V,xn as W,J1 as X,gf as Y,_s as Z,Ts as _,qt as a,za as a$,to as a0,Q1 as a1,$1 as a2,Ds as a3,Se as a4,un as a5,_r as a6,vs as a7,Io as a8,Vt as a9,Ta as aA,_a as aB,t1 as aC,Na as aD,Qw as aE,Da as aF,Aa as aG,Ma as aH,yo as aI,e1 as aJ,pr as aK,Ra as aL,Pa as aM,Ot as aN,ln as aO,co as aP,us as aQ,La as aR,Ua as aS,Ge as aT,Ke as aU,Wa as aV,Qr as aW,Mm as aX,Ka as aY,$n as aZ,n1 as a_,lb as aa,hb as ab,fb as ac,fm as ad,ma as ae,S as af,_o as ag,ba as ah,Et as ai,Mt as aj,R as ak,U as al,_t as am,z as an,wa as ao,dr as ap,K as aq,x as ar,ya as as,ka as at,mt as au,va as av,Ea as aw,Sa as ax,M as ay,Ia as az,Wt as b,rc as b$,jh as b0,Ha as b1,kg as b2,Va as b3,Eg as b4,Xa as b5,rf as b6,Qa as b7,Om as b8,Rm as b9,c1 as bA,l1 as bB,Ri as bC,eo as bD,Pi as bE,Oi as bF,Yn as bG,d1 as bH,Ui as bI,f1 as bJ,Li as bK,qi as bL,Zt as bM,Ki as bN,zi as bO,Gi as bP,j as bQ,Hi as bR,ao as bS,ji as bT,Xi as bU,ec as bV,Te as bW,tc as bX,nc as bY,Es as bZ,sc as b_,si as ba,s1 as bb,r1 as bc,ai as bd,a1 as be,ii as bf,ee as bg,ui as bh,li as bi,hi as bj,gi as bk,mi as bl,bi as bm,fg as bn,Me as bo,wi as bp,cm as bq,xi as br,gr as bs,Ei as bt,Si as bu,Ti as bv,_i as bw,vn as bx,Mi as by,Ai as bz,pn as c,Wl as c$,Le as c0,Ue as c1,oc as c2,ac as c3,nf as c4,ri as c5,fc as c6,bc as c7,dc as c8,pc as c9,Mc as cA,b1 as cB,qc as cC,su as cD,jc as cE,Fc as cF,Xc as cG,Yc as cH,mr as cI,In as cJ,tu as cK,ze as cL,eu as cM,bo as cN,io as cO,Ht as cP,nu as cQ,x1 as cR,_u as cS,gt as cT,Sl as cU,_l as cV,Al as cW,Fl as cX,Cl as cY,Pl as cZ,Ll as c_,mc as ca,m1 as cb,gc as cc,g1 as cd,wc as ce,ne as cf,yc as cg,$c as ch,Ec as ci,po as cj,Sc as ck,Nc as cl,Dc as cm,_c as cn,Jh as co,Ic as cp,tf as cq,Tc as cr,Ze as cs,Rc as ct,Ac as cu,we as cv,Bc as cw,Zr as cx,Cc as cy,ft as cz,f as d,Vg as d$,Kl as d0,Gl as d1,Vl as d2,Yr as d3,yn as d4,nn as d5,Ih as d6,Nh as d7,Wh as d8,zh as d9,ye as dA,hn as dB,Xn as dC,fn as dD,Pd as dE,Ld as dF,kn as dG,wo as dH,Yd as dI,fp as dJ,$o as dK,gp as dL,Jp as dM,Tn as dN,vo as dO,Tb as dP,Ib as dQ,xs as dR,Eo as dS,pg as dT,mg as dU,$g as dV,Bg as dW,ho as dX,To as dY,ks as dZ,Gg as d_,uf as da,cs as db,pf as dc,$f as dd,kf as de,no as df,Df as dg,Lf as dh,Kf as di,$s as dj,jr as dk,dn as dl,So as dm,Jf as dn,td as dp,nd as dq,uo as dr,cd as ds,gd as dt,wd as du,fo as dv,lo as dw,go as dx,Ed as dy,mo as dz,rs as e,bs as e$,jn as e0,nm as e1,am as e2,Ce as e3,G1 as e4,J as e5,Mg as e6,Ng as e7,Ig as e8,Tg as e9,qu as eA,Kw as eB,Yu as eC,Vw as eD,Rr as eE,Yh as eF,vw as eG,Bm as eH,Hh as eI,Wd as eJ,mh as eK,wh as eL,$h as eM,Dd as eN,rh as eO,ge as eP,We as eQ,af as eR,pe as eS,c0 as eT,mn as eU,i0 as eV,Lr as eW,Or as eX,Uo as eY,os as eZ,Du as e_,ko as ea,qm as eb,vt as ec,Oh as ed,Rh as ee,Bh as ef,Mh as eg,Im as eh,Bo as ei,ys as ej,rm as ek,oo as el,Ew as em,Fo as en,Ct as eo,hm as ep,ht as eq,C1 as er,ur as es,wl as et,Ye as eu,Jt as ev,ce as ew,er as ex,Xu as ey,Ou as ez,ve as f,Qi as f$,Fe as f0,bu as f1,Go as f2,Jw as f3,B1 as f4,aa as f5,At as f6,rr as f7,ua as f8,ca as f9,Ga as fA,Ya as fB,ja as fC,ts as fD,Za as fE,Ja as fF,ti as fG,ei as fH,ni as fI,ia as fJ,oi as fK,fi as fL,di as fM,He as fN,pi as fO,Cs as fP,Rs as fQ,yi as fR,ki as fS,Ni as fT,Fi as fU,Bi as fV,Ci as fW,Wi as fX,Vi as fY,Zi as fZ,Ji as f_,Oa as fa,gn as fb,hc as fc,bn as fd,ra as fe,Ba as ff,Qn as fg,ci as fh,$i as fi,Ii as fj,Di as fk,gu as fl,Yi as fm,Pn as fn,Kc as fo,pu as fp,sn as fq,z1 as fr,sa as fs,Bs as ft,$a as fu,xa as fv,Fa as fw,Ca as fx,vi as fy,qa as fz,p as g,Ef as g$,at as g0,ic as g1,cc as g2,uc as g3,lc as g4,ru as g5,xc as g6,vc as g7,Pc as g8,Oc as g9,_1 as gA,o0 as gB,I1 as gC,an as gD,Hs as gE,it as gF,Ft as gG,e0 as gH,n0 as gI,s0 as gJ,r0 as gK,hd as gL,la as gM,hr as gN,Sw as gO,xw as gP,Km as gQ,kw as gR,Tw as gS,_w as gT,Dl as gU,hh as gV,Jr as gW,vh as gX,Sh as gY,Ut as gZ,ff as g_,Lc as ga,Uc as gb,Wc as gc,zc as gd,Gc as ge,Hc as gf,Vc as gg,kc as gh,Zc as gi,Jc as gj,Qc as gk,ou as gl,ta as gm,Ae as gn,jo as go,sr as gp,Cu as gq,na as gr,Xo as gs,rn as gt,Fr as gu,$e as gv,T1 as gw,a0 as gx,S1 as gy,X1 as gz,jd as h,u1 as h$,_f as h0,En as h1,ad as h2,Td as h3,Ad as h4,Bd as h5,Kd as h6,Gd as h7,Jd as h8,ep as h9,$m as hA,ls as hB,vm as hC,Sm as hD,No as hE,Ss as hF,Nm as hG,D1 as hH,N1 as hI,A1 as hJ,M1 as hK,F1 as hL,R1 as hM,P1 as hN,O1 as hO,L1 as hP,U1 as hQ,W1 as hR,q1 as hS,K1 as hT,H1 as hU,Os as hV,Fn as hW,k1 as hX,v1 as hY,E1 as hZ,i1 as h_,sp as ha,op as hb,ip as hc,Vr as hd,bp as he,yp as hf,xp as hg,vp as hh,Kp as hi,Hp as hj,Xp as hk,qe as hl,sg as hm,og as hn,ig as ho,ug as hp,wg as hq,Xg as hr,Yg as hs,Zg as ht,tm as hu,lm as hv,Do as hw,Cr as hx,gm as hy,wm as hz,bt as i,h1 as i0,p1 as i1,w1 as i2,st as j,B as k,Ps as l,O as m,jg as n,m as o,Ve as p,Vo as q,Iw as r,Z as s,Zn as t,Aw as u,Dw as v,oa as w,Rn as x,Is as y,Q as z};
