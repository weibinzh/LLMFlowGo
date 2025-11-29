import React, { useEffect, useState } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { Card, Form, InputNumber, Row, Col, Button, message, Typography } from 'antd';
import { runsApi } from '../services/api';

const { Title, Text } = Typography;

const Planner = () => {
  const { runId } = useParams();
  const navigate = useNavigate();
  const [form] = Form.useForm();
  const [loading, setLoading] = useState(false);
  const [counts, setCounts] = useState({ cloud: 1, edge: 1, device: 1 });

  const fetchResults = async () => {
    if (!runId) {
      message.error('Missing runId parameter');
      return;
    }
    setLoading(true);
    try {
      const resp = await runsApi.getResults(runId);
      const payload = resp.data;
      const c = payload?.counts || {};
      const nextCounts = {
        cloud: Number(c.cloud ?? 1),
        edge: Number(c.edge ?? 1),
        device: Number(c.device ?? 1),
      };
      setCounts(nextCounts);
      form.setFieldsValue({
        cloudCount: nextCounts.cloud,
        edgeCount: nextCounts.edge,
        deviceCount: nextCounts.device,
      });
    } catch (e) {
      console.error(e);
      message.error(e?.response?.data?.detail || 'Failed to fetch run results');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchResults();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [runId]);

  const goToFinalAnalysis = () => {
    navigate(`/analysis/${runId}`);
  };

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
                        <Text strong>Medium</Text>
                        <div style={{ fontSize: '12px', color: '#666' }}>3200 MIPS | $1.66/s | 200 Mbps</div>
                      </div>
                    </Form.Item>
                  </Col>
                  <Col span={12}>
                    <Form.Item label="Count" name="cloudCount" rules={[{ required: true, message: 'Please enter cloud server count' }]}> 
                      <InputNumber
                        placeholder="Enter count"
                        style={{ width: '100%' }}
                        min={1}
                        step={1}
                        precision={0}
                        value={counts.cloud}
                        onChange={(value) => setCounts((prev) => ({ ...prev, cloud: Math.max(1, Math.round(Number(value ?? 1))) }))}
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
                    <Form.Item label="Count" name="edgeCount" rules={[{ required: true, message: 'Please enter edge server count' }]}> 
                      <InputNumber
                        placeholder="Enter count"
                        style={{ width: '100%' }}
                        min={1}
                        step={1}
                        precision={0}
                        value={counts.edge}
                        onChange={(value) => setCounts((prev) => ({ ...prev, edge: Math.max(1, Math.round(Number(value ?? 1))) }))}
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
                        <Text strong>Medium</Text>
                        <div style={{ fontSize: '12px', color: '#666' }}>2000 MIPS | $0/s | 20 Mbps</div>
                      </div>
                    </Form.Item>
                  </Col>
                  <Col span={12}>
                    <Form.Item label="Count" name="deviceCount" rules={[{ required: true, message: 'Please enter device count' }]}> 
                      <InputNumber
                        placeholder="Enter count"
                        style={{ width: '100%' }}
                        min={1}
                        step={1}
                        precision={0}
                        value={counts.device}
                        onChange={(value) => setCounts((prev) => ({ ...prev, device: Math.max(1, Math.round(Number(value ?? 1))) }))}
                      />
                    </Form.Item>
                  </Col>
                </Row>
              </Card>

              <div style={{ display: 'flex', gap: 8 }}>
                <Button type="primary" loading={loading} onClick={fetchResults}>Refresh Run Results</Button>
                <Button onClick={goToFinalAnalysis}>View Final Results</Button>
              </div>
            </Form>
          </Col>
        </Row>
      </Card>
    </div>
  );
};

export default Planner;
