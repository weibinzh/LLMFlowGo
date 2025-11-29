import React, { useEffect, useState } from 'react';
import { Card, Button, Form, InputNumber, Select, message, Row, Col, Typography, Divider, Alert, Switch, Input, Radio, Checkbox } from 'antd';
import { useLocation, useNavigate } from 'react-router-dom';
import { problemsApi } from '../services/api';
import useLLMConfigStore from '../store/llmConfigStore';

const { Title, Text } = Typography;
const { Option } = Select;

const EnvironmentSetup = () => {
  const navigate = useNavigate();
  const location = useLocation();
  const [form] = Form.useForm();
  const [environmentConfig, setEnvironmentConfig] = useState({
    cloudSpec: 'cloud-large',
    cloudCount: 1,
    edgeSpec: 'edge-medium',
    edgeCount: 1,
    deviceSpec: 'device-small',
    deviceCount: 1,
  });

  const [dagConfig, setDagConfig] = useState(() => location.state?.dagConfig || null);

  // Retrieve LLM config from global store
  const { apiKey, baseUrl, modelName } = useLLMConfigStore();
  const llmConfig = apiKey && baseUrl && modelName ? { apiKey, baseUrl, modelName } : null;

  // Local state: save pendingRunId to trigger re-render
  const [pendingRunId, setPendingRunId] = useState(null);
  const [presets, setPresets] = useState([]);
  const [recommendedPreset, setRecommendedPreset] = useState(null);
  const [recommendedReason, setRecommendedReason] = useState('');
  const [adoptRecommended, setAdoptRecommended] = useState(false);
  const [algorithmPreset, setAlgorithmPreset] = useState(null);
  const [optReason, setOptReason] = useState('');
  const [optDraft, setOptDraft] = useState('');
  const [analysisList, setAnalysisList] = useState([]);
  const [optSuggestion, setOptSuggestion] = useState('');

  // Resolve current runId (state → localStorage.pendingRunId → localStorage.autoRunPayload.runId)
  const resolveRunId = () => {
    let id = pendingRunId;
    if (!id) {
      try { id = localStorage.getItem('pendingRunId') || null; } catch {}
    }
    if (!id) {
      try {
        const arpRaw = localStorage.getItem('autoRunPayload');
        if (arpRaw) {
          const arp = JSON.parse(arpRaw);
          if (arp?.runId) id = arp.runId;
        }
      } catch {}
    }
    return id;
  };

  const handleGoToRunPage = () => {
    const id = resolveRunId();
    if (!id) {
      message.info('No run to navigate. Please create an Edge Workflow and start a run first.');
      return;
    }
    navigate(`/planner/${id}`);
  };
  // Recommended ranges (from LLM); used to set min/max for counts
  const [recommendedRanges, setRecommendedRanges] = useState({
    cloud: null,
    edge: null,
    device: null,
  });
  const [recommendationsReady, setRecommendationsReady] = useState(false);
  const [isRefreshing, setIsRefreshing] = useState(false);

  // Three server types config options: fixed spec
  const serverTypes = {
    cloud: {
      name: 'Cloud Servers',
      description: 'High-performance cloud compute resources for complex tasks',
      icon: '☁️',
      specs: [
        { value: 'cloud-medium', label: 'Medium', mips: 3200, price: 1.66, bandwidth: 200 },
      ],
    },
    edge: {
      name: 'Edge Servers',
      description: 'Compute nodes at the network edge offering low-latency processing',
      icon: '🌐',
      specs: [
        { value: 'edge-medium', label: 'Medium', mips: 2600, price: 0.78, bandwidth: 100 },
      ],
    },
    device: {
      name: 'Devices',
      description: 'Endpoint compute resources for lightweight tasks and data collection',
      icon: '📱',
      specs: [
        { value: 'device-medium', label: 'Medium', mips: 2000, price: 0, bandwidth: 20 },
      ],
    },
  };

  const refreshRecommendations = async (fingerprint = null) => {
    // Only used to clean up concurrency flags in the finally block; do not make early stopping decisions here.
    const inflightKey = fingerprint ? ('rec_inflight:' + fingerprint) : null;

    if (isRefreshing) {
      return;
    }
    if (!llmConfig) {
      message.error('Missing LLM settings. Please configure them in LLM Settings (top-right).');
      // Clean up the in-flight flag to avoid deadlocks
      try { if (inflightKey) sessionStorage.removeItem(inflightKey); } catch (e) {}
      return;
    }
    setIsRefreshing(true);
    let hide;
    try {
      hide = message.loading('Calculating recommended ranges...', 0);
      const resp = await problemsApi.recommendServerCounts(dagConfig, serverTypes, llmConfig, null);
      if (resp.data && resp.data.success) {
        const rec = resp.data.recommendation || {};
        const cloudRange = rec.cloudRange || (rec.cloudCount != null ? [rec.cloudCount, rec.cloudCount] : null);
        const edgeRange = rec.edgeRange || (rec.edgeCount != null ? [rec.edgeCount, rec.edgeCount] : null);
        const deviceRange = rec.deviceRange || (rec.deviceCount != null ? [rec.deviceCount, rec.deviceCount] : null);
        if (!cloudRange || !edgeRange || !deviceRange) {
          throw new Error('LLM returned incomplete results');
        }
        setRecommendedRanges({ cloud: cloudRange, edge: edgeRange, device: deviceRange });
        const mid = (arr) => Math.round((arr[0] + arr[1]) / 2);
        const updated = {
          ...environmentConfig,
          cloudCount: mid(cloudRange),
          edgeCount: mid(edgeRange),
          deviceCount: mid(deviceRange),
        };
        setEnvironmentConfig(updated);
        form.setFieldsValue({
          cloudSpec: updated.cloudSpec,
          edgeSpec: updated.edgeSpec,
          deviceSpec: updated.deviceSpec,
          cloudCount: updated.cloudCount,
          edgeCount: updated.edgeCount,
          deviceCount: updated.deviceCount,
        });
        setRecommendationsReady(true);
        const algoPreset = (resp.data && resp.data.algoPreset) ? resp.data.algoPreset : (rec && rec.algoPreset ? rec.algoPreset : null);
        const algoReason = (resp.data && resp.data.algoReason) ? resp.data.algoReason : (rec && rec.algoReason ? rec.algoReason : '');
        if (algoPreset) {
          setRecommendedPreset(algoPreset);
          setRecommendedReason(algoReason);
          setAdoptRecommended(true);
          setAlgorithmPreset(algoPreset);
        }
        message.success(`Applied recommended ranges: Cloud ${cloudRange[0]}–${cloudRange[1]}, Edge ${edgeRange[0]}–${edgeRange[1]}, Devices ${deviceRange[0]}–${deviceRange[1]}`);
      } else {
        throw new Error('LLM did not return recommendations');
      }
    } catch (err) {
      console.error(err);
      setRecommendationsReady(false);
      message.error('LLM recommendation failed. Check LLM settings or try again later.');
    } finally {
      if (hide) hide();
      setIsRefreshing(false);
      // Clean up the in-flight flag after the request ends (do not set the completion flag)
      try { if (inflightKey) sessionStorage.removeItem(inflightKey); } catch (e) {}
    }
  };

  useEffect(() => {
    // Set the fixed specs in the form initially 
    form.setFieldsValue({
      cloudSpec: environmentConfig.cloudSpec,
      edgeSpec: environmentConfig.edgeSpec,
      deviceSpec: environmentConfig.deviceSpec,
    });

    // Set the default values for MEOH parameters
    try {
      form.setFieldsValue({
        pop_size: 20,
        max_generations: 10,
        max_sample_nums: 100,
        selection_num: 4,
        num_samplers: 4,
        num_evaluators: 4,
        use_e2_operator: true,
        use_m1_operator: true,
        use_m2_operator: true,
      });
    } catch (e) {}
  }, []);

  // When a new recommendation algorithm appears, automatically select the corresponding radio button.
  useEffect(() => {
    if (recommendedPreset && !algorithmPreset) {
      setAlgorithmPreset(recommendedPreset);
    }
  }, [recommendedPreset]);

  // Fallback: Restore dagConfig from localStorage (used to automatically return from RunMonitoring)
  useEffect(() => {
    if (!dagConfig) {
      const storedDag = localStorage.getItem('dagConfig');
      if (storedDag) {
        try {
          const parsed = JSON.parse(storedDag);
          setDagConfig(parsed);
        } catch (e) {
          console.warn('Failed to parse dagConfig from localStorage');
        }
      }
    }
  }, [dagConfig]);

  // Automatically apply precise counts (from RunMonitoring saved preciseCounts)
  useEffect(() => {
    const storedCounts = localStorage.getItem('preciseCounts');
    if (storedCounts) {
      try {
        const { cloud, edge, device } = JSON.parse(storedCounts);
        const updated = {
          ...environmentConfig,
          cloudCount: cloud,
          edgeCount: edge,
          deviceCount: device,
        };
        setEnvironmentConfig(updated);
        form.setFieldsValue({
          cloudSpec: updated.cloudSpec,
          edgeSpec: updated.edgeSpec,
          deviceSpec: updated.deviceSpec,
          cloudCount: updated.cloudCount,
          edgeCount: updated.edgeCount,
          deviceCount: updated.deviceCount,
        });
        setRecommendedRanges({ cloud: [cloud, cloud], edge: [edge, edge], device: [device, device] });
        setRecommendationsReady(true);
        message.success(`Applied precise counts: Cloud ${cloud}, Edge ${edge}, Devices ${device}`);
      } catch (e) {
        console.warn('Failed to parse preciseCounts from localStorage');
      } finally {
        localStorage.removeItem('preciseCounts');
      }
    }
  }, []);

  useEffect(() => {
    if (dagConfig && dagConfig.workload && Object.keys(dagConfig.workload).length > 0 && dagConfig.edges && dagConfig.edges.length > 0) {
       // Triggered each time the page is entered; use the in-flight flag to avoid concurrent requests in StrictMode
       const fp = (() => { try { return JSON.stringify(dagConfig); } catch (e) { return null; } })();
       const inflightKey = fp ? ('rec_inflight:' + fp) : null;
       try {
         if (inflightKey && sessionStorage.getItem(inflightKey)) {
           return;
         }
         if (inflightKey) sessionStorage.setItem(inflightKey, '1');
       } catch (e) {}
       refreshRecommendations(fp || null);
    }
  }, [dagConfig]);

  const goToDagSetup = () => {
    navigate('/workflow');
  };

  if (!dagConfig || !dagConfig.edges || dagConfig.edges.length === 0) {
    return (
      <div>
        <Title level={2}>Server Environment Setup</Title>
        <Alert
          message="DAG setup not completed"
          description="Please complete the DAG workflow configuration first, then proceed to Server Environment Setup."
          type="warning"
          showIcon
          style={{ marginBottom: 16 }}
        />
        <Button type="primary" onClick={goToDagSetup}>Go to DAG Setup</Button>
      </div>
    );
  }

  return (
    <div>
      <Title level={2}>Server Environment Setup</Title>
      <Card title="Server Environment Setup">
        <Row gutter={16}>
          <Col span={12}>
            <Form form={form} layout="vertical">
              
              <Card title="☁️ Cloud Server Configuration" size="small" style={{ marginBottom: 16 }}>
                <Row gutter={16}>
                  <Col span={12}>
                    <Form.Item label="Spec" name="cloudSpec">
                      <div>
                        <Text strong>Large</Text>
                        <div style={{ fontSize: '12px', color: '#666' }}>4800 MIPS | $2.36/s | 200 Mbps</div>
                      </div>
                    </Form.Item>
                  </Col>
                  <Col span={12}>
                    <Form.Item
                      label="Count"
                      name="cloudCount"
                      rules={[{ required: true, message: 'Please enter cloud server count' }]}
                      extra={<Text type="secondary">Recommended range: {recommendationsReady && recommendedRanges.cloud ? `${recommendedRanges.cloud[0]}–${recommendedRanges.cloud[1]}` : 'Not available'}</Text>}
                    >
                      <InputNumber
                        min={recommendationsReady && recommendedRanges.cloud ? recommendedRanges.cloud[0] : undefined}
                        max={recommendationsReady && recommendedRanges.cloud ? recommendedRanges.cloud[1] : undefined}
                        placeholder="Enter count"
                        style={{ width: '100%' }}
                        disabled={!recommendationsReady}
                        step={1}
                        precision={0}
                        onChange={(value) => setEnvironmentConfig((prev) => ({ ...prev, cloudCount: Math.round(Number(value ?? 1)) }))}
                      />
                    </Form.Item>
                  </Col>
                </Row>
              </Card>

              <Card title="🌐 Edge Server Configuration" size="small" style={{ marginBottom: 16 }}>
                <Row gutter={16}>
                  <Col span={12}>
                    <Form.Item label="Spec" name="edgeSpec">
                      <div>
                        <Text strong>Medium</Text>
                        <div style={{ fontSize: '12px', color: '#666' }}>2600 MIPS | $0.78/s | 100 Mbps</div>
                      </div>
                    </Form.Item>
                  </Col>
                  <Col span={12}>
                    <Form.Item
                      label="Count"
                      name="edgeCount"
                      rules={[{ required: true, message: 'Please enter edge server count' }]}
                      extra={<Text type="secondary">Recommended range: {recommendationsReady && recommendedRanges.edge ? `${recommendedRanges.edge[0]}–${recommendedRanges.edge[1]}` : 'Not available'}</Text>}
                    >
                      <InputNumber
                        min={recommendationsReady && recommendedRanges.edge ? recommendedRanges.edge[0] : undefined}
                        max={recommendationsReady && recommendedRanges.edge ? recommendedRanges.edge[1] : undefined}
                        placeholder="Enter count"
                        style={{ width: '100%' }}
                        disabled={!recommendationsReady}
                        step={1}
                        precision={0}
                        onChange={(value) => setEnvironmentConfig((prev) => ({ ...prev, edgeCount: Math.round(Number(value ?? 1)) }))}
                      />
                    </Form.Item>
                  </Col>
                </Row>
              </Card>

              <Card title="📱 Device Configuration" size="small" style={{ marginBottom: 16 }}>
                <Row gutter={16}>
                  <Col span={12}>
                    <Form.Item label="Spec" name="deviceSpec">
                      <div>
                        <Text strong>Small</Text>
                        <div style={{ fontSize: '12px', color: '#666' }}>1000 MIPS | $0/s | 20 Mbps</div>
                      </div>
                    </Form.Item>
                  </Col>
                  <Col span={12}>
                    <Form.Item
                      label="Count"
                      name="deviceCount"
                      rules={[{ required: true, message: 'Please enter device count' }]}
                      extra={<Text type="secondary">Recommended range: {recommendationsReady && recommendedRanges.device ? `${recommendedRanges.device[0]}–${recommendedRanges.device[1]}` : 'Not available'}</Text>}
                    >
                      <InputNumber
                        min={recommendationsReady && recommendedRanges.device ? recommendedRanges.device[0] : undefined}
                        max={recommendationsReady && recommendedRanges.device ? recommendedRanges.device[1] : undefined}
                        placeholder="Enter count"
                        style={{ width: '100%' }}
                        disabled={!recommendationsReady}
                        step={1}
                        precision={0}
                        onChange={(value) => setEnvironmentConfig((prev) => ({ ...prev, deviceCount: Math.round(Number(value ?? 1)) }))}
                      />
                    </Form.Item>
                  </Col>
                </Row>
              </Card>

              <Card title="Algorithm Advantage (LLM response)" size="small" style={{ marginBottom: 16 }}>
                {recommendedReason ? (
                  <Alert type="info" showIcon message={`Recommended reason: ${recommendedReason}`} style={{ marginBottom: 8 }} />
                ) : (
                  <Alert type="warning" showIcon message={`No recommended reason`} style={{ marginBottom: 8 }} />
                )}
              </Card>

              <Card title="Algorithm Recommendation & Selection" size="small" style={{ marginBottom: 16 }}>
                <Radio.Group
                  style={{ width: '100%' }}
                  value={algorithmPreset}
                  onChange={(e) => setAlgorithmPreset(e.target.value)}
                >
                  <Radio value="pso" style={{ display: 'block', lineHeight: '28px' }}>PSO (Particle Swarm)</Radio>
                  <Radio value="ga" style={{ display: 'block', lineHeight: '28px' }}>GA (Genetic Algorithm)</Radio>
                  <Radio value="local search" style={{ display: 'block', lineHeight: '28px' }}>Local Search</Radio>
                </Radio.Group>
                <div style={{ textAlign: 'right', marginTop: 12 }}>
                  <Button
                    type="primary"
                    disabled={!algorithmPreset || !llmConfig}
                    onClick={async () => {
                      try {
                        const resp = await problemsApi.analyzePresetFramework(algorithmPreset, dagConfig, environmentConfig, serverTypes, llmConfig);
                        const rsn = (resp.data && resp.data.reason) ? resp.data.reason : '';
                        const dsc = (resp.data && resp.data.description) ? resp.data.description : '';
                        setOptReason(rsn);
                        setRecommendedReason(rsn);
                        setOptSuggestion(dsc);
                        if (!optDraft || !String(optDraft).trim()) {
                          if (dsc && String(dsc).trim()) {
                            setOptDraft(dsc);
                          }
                        }
                        message.success('LLM reason generated and filled');
                      } catch (e) {
                        message.error('Algorithm framework analysis failed');
                      }
                    }}
                  >
                    Confirm Algorithm
                  </Button>
                </div>
              </Card>

              <Card title="Optimization Direction (LLM response)" size="small" style={{ marginBottom: 16 }}>
                {optReason ? (
                  <Alert type="info" showIcon message={`LLM Reason: ${optReason}`} style={{ marginBottom: 8 }} />
                ) : null}
                <div style={{ whiteSpace: 'pre-wrap' }}>{optSuggestion || ''}</div>
              </Card>

              <Card title="Optimization Direction Editor" size="small" style={{ marginBottom: 16 }}>
                <Input.TextArea
                  rows={6}
                  value={optDraft}
                  onChange={(e) => setOptDraft(e.target.value)}
                  placeholder="Add your additional description; it will be appended to the system prompt"
                />
              </Card>

              {false && (<Card title="Optimization Direction Editor" size="small" style={{ marginBottom: 16 }}>
                {optReason ? (
                  <Alert type="info" showIcon message={`LLM Reason: ${optReason}`} style={{ marginBottom: 8 }} />
                ) : null}
                <Input.TextArea
                  rows={6}
                  value={optDraft}
                  onChange={(e) => setOptDraft(e.target.value)}
                  placeholder="Add your additional description; it will be appended to the system prompt"
                />
              </Card>)}

              <div style={{ textAlign: 'right', marginBottom: 16 }}>
                <Button
                  type="primary"
                    title={!llmConfig ? 'Please configure LLM settings in the top-right first' : ''}
                  onClick={async () => {
                    if (!llmConfig) {
                      message.warning('Please configure LLM settings in the top-right first');
                      return;
                    }
                    try {
                      await form.validateFields();
                    } catch (e) {
                      message.warning('Please complete count configuration first');
                      return;
                    }
                    const hide = message.loading('Creating and starting Precise run...', 0);
                    try {
                      const meohConfig = {
                        pop_size: Number(form.getFieldValue('pop_size')),
                        max_generations: Number(form.getFieldValue('max_generations')),
                        max_sample_nums: Number(form.getFieldValue('max_sample_nums')),
                        selection_num: Number(form.getFieldValue('selection_num')),
                        use_e2_operator: !!form.getFieldValue('use_e2_operator'),
                        use_m1_operator: !!form.getFieldValue('use_m1_operator'),
                        use_m2_operator: !!form.getFieldValue('use_m2_operator'),
                        num_samplers: Number(form.getFieldValue('num_samplers')),
                        num_evaluators: Number(form.getFieldValue('num_evaluators')),
                      };
                    const combinedDescription = (() => {
                      const parts = [];
                      const suggestion = (optSuggestion && String(optSuggestion).trim()) ? String(optSuggestion).trim() : '';
                      if (suggestion) parts.push(suggestion);
                      const draft = (optDraft && String(optDraft).trim()) ? String(optDraft).trim() : '';
                      if (draft) parts.push(draft);
                      return parts.join('\n\n');
                    })();
                    const resp = await problemsApi.preciseBuildRun(
                        environmentConfig,
                        dagConfig,
                        llmConfig,
                        null,
                        combinedDescription || null,
                        meohConfig,
                        recommendationsReady ? recommendedRanges : null,
                        algorithmPreset || null
                      );
                      hide();
                      if (resp.data && resp.data.problem) {
                        try {
                          localStorage.setItem('dagConfig', JSON.stringify(dagConfig));
                        } catch (e) {
                          console.warn('Failed to save dagConfig to localStorage');
                        }

                        const meohConfigReturned = resp.data.run && resp.data.run.meoh_config ? resp.data.run.meoh_config : meohConfig;
                        const runId = resp.data.run && resp.data.run.id ? resp.data.run.id : null;
                        try {
                          localStorage.setItem('autoRunPayload', JSON.stringify({
                            problemId: resp.data.problem.id,
                            runId,
                            config: meohConfigReturned,
                          }));
                          if (runId) {
                            localStorage.setItem('pendingRunId', runId);
                            setPendingRunId(runId);
                          }
                        } catch (e) {
                          console.warn('Failed to write autoRunPayload', e);
                        }

                        message.success(`Created Edge Workflow ${resp.data.problem.id} and started a run (no auto navigation).`);
                      } else {
                        message.warning('Create or start failed: incomplete response');
                      }
                    } catch (err) {
                      hide();
                      message.error(err?.response?.data?.detail || 'Failed to create and start Precise run');
                    }
                  }}
                >
                  Next: Create Edge Workflow
                </Button>
              </div>

              {/* */}
              <Card title="⚙️ MEOH Parameters" size="small" style={{ marginBottom: 16 }}>
                <Row gutter={16}>
                  <Col span={12}>
                    <Form.Item label="Population Size" name="pop_size" rules={[{ required: true, message: 'Please enter population size' }]}> 
                      <InputNumber min={1} style={{ width: '100%' }} />
                    </Form.Item>
                  </Col>
                  <Col span={12}>
                    <Form.Item label="Max Generations" name="max_generations" rules={[{ required: true, message: 'Please enter max generations' }]}> 
                      <InputNumber min={1} style={{ width: '100%' }} />
                    </Form.Item>
                  </Col>
                </Row>
                <Row gutter={16}>
                  <Col span={12}>
                    <Form.Item label="Max Sample Numbers" name="max_sample_nums" rules={[{ required: true, message: 'Please enter max sample numbers' }]}> 
                      <InputNumber min={1} style={{ width: '100%' }} />
                    </Form.Item>
                  </Col>
                  <Col span={12}>
                    <Form.Item label="Selection Number" name="selection_num" rules={[{ required: true, message: 'Please enter selection number' }]}> 
                      <InputNumber min={1} style={{ width: '100%' }} />
                    </Form.Item>
                  </Col>
                </Row>
                <Row gutter={16}>
                  <Col span={12}>
                    <Form.Item label="Number of Samplers" name="num_samplers" rules={[{ required: true, message: 'Please enter number of samplers' }]}> 
                      <InputNumber min={1} style={{ width: '100%' }} />
                    </Form.Item>
                  </Col>
                  <Col span={12}>
                    <Form.Item label="Number of Evaluators" name="num_evaluators" rules={[{ required: true, message: 'Please enter number of evaluators' }]}> 
                      <InputNumber min={1} style={{ width: '100%' }} />
                    </Form.Item>
                  </Col>
                </Row>
                <Row gutter={16}>
                  <Col span={8}>
                    <Form.Item label="use_e2_operator" name="use_e2_operator" valuePropName="checked">
                      <Switch />
                    </Form.Item>
                  </Col>
                  <Col span={8}>
                    <Form.Item label="use_m1_operator" name="use_m1_operator" valuePropName="checked">
                      <Switch />
                    </Form.Item>
                  </Col>
                  <Col span={8}>
                    <Form.Item label="use_m2_operator" name="use_m2_operator" valuePropName="checked">
                      <Switch />
                    </Form.Item>
                  </Col>
                </Row>
                <Divider />
                <div style={{ textAlign: 'left', marginBottom: 8 }}>
                  <Text>Current Run ID: {resolveRunId() || 'Not available'}</Text>
                </div>
                <div style={{ textAlign: 'right' }}>
                  <Button
                    style={{ marginLeft: 8 }}
                    onClick={handleGoToRunPage}
                    disabled={!resolveRunId()}
                    title={!resolveRunId() ? 'Please create an Edge Workflow and start a run first' : `Go to run ${resolveRunId()}`}
                  >
                    Go to Run Page
                  </Button>
                </div>
              </Card>
            </Form>
          </Col>
        </Row>
      </Card>
    </div>
  );
};

export default EnvironmentSetup;
