# CLAUDE.md - cách làm việc trong repo vi-rag-agent

Repo portfolio của Dương Minh Quang (xin thực tập/fresher AI Engineer). Tiếng Việt khi trao đổi; code, identifier,
commit, README, docs trong repo bằng tiếng Anh. Kế hoạch: `docs/ROADMAP.md`. Quy trình đánh giá: `docs/EVALUATION.md`.

## Lệnh thường dùng
```powershell
pip install -e ".[dev]"
ruff check . ; ruff format --check . ; pytest
```

## Bất biến
1. Không bịa số liệu. Số trong README/CHANGELOG chỉ lấy từ `results/*.json` của lần chạy thật. Chưa đo thì ghi "not measured".
2. Không rò rỉ dữ liệu: chia train/dev/test một lần, cố định seed, chỉ tune trên dev.
3. Không commit dữ liệu, `.env`, token, checkpoint, index. Không dùng nhãn đáp án công khai để "tái hiện" kết quả.
4. Mọi hàm đo (metric) có test so với giá trị tính tay. Sửa logic thì chạy lại test.
5. Không sao chép code người khác; ghi nguồn và license của mọi dataset/model.
6. Nói rõ điều gì chưa kiểm chứng.

## Chất lượng: không mặc định là đúng
Trước khi báo "xong" một việc, Claude phải làm đủ các bước sau và ghi kết quả trong PR:
1. **Recheck**: chạy lại lint, test; tính lại mọi con số từ file kết quả; đọc lại diff như một reviewer.
2. **Phản biện**: nêu ít nhất 3 câu hỏi khó mà người phỏng vấn/reviewer có thể hỏi (rò rỉ dữ liệu, baseline, cỡ mẫu, so sánh có công bằng không) và trả lời bằng bằng chứng hoặc ghi "chưa kiểm chứng".
3. **Thử phá**: với code mới, thử đột biến (mutation) hoặc ca biên để xem test có thật sự bắt được lỗi.
4. **Gợi ý**: đề xuất ít nhất 2 hướng cải thiện kèm chi phí/lợi ích; nêu lựa chọn thay thế đã cân nhắc.
5. **Không đồng ý khi cần**: nếu yêu cầu của người dùng có rủi ro hoặc có cách tốt hơn, nói rõ trước khi làm.

## Quy trình GitHub
Milestone -> issue (template Task/Experiment) -> nhánh `feat/...` -> PR (`Closes #n`, điền đủ template) -> review -> squash-merge.
Đóng milestone: viết `docs/reports/vX.Y.0.md`, chuyển CHANGELOG, mở PR `chore(release): vX.Y.Z`; workflow `release` tự tạo tag + Release (xem `docs/reports/README.md`).
Commit theo Conventional Commits. Không push thẳng vào `main`. Lịch sử git trung thực, không backdate.
